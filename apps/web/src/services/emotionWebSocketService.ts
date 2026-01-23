/**
 * 情感智能WebSocket客户端服务
 * 提供实时情感数据传输和处理功能
 */

import { EventEmitter } from 'events'

import { logger } from '../utils/logger'
// 消息类型枚举
export enum MessageType {
  CONNECT = 'connect',
  DISCONNECT = 'disconnect',
  EMOTION_INPUT = 'emotion_input',
  EMOTION_RESULT = 'emotion_result',
  STREAM_STATUS = 'stream_status',
  ERROR = 'error',
  HEARTBEAT = 'heartbeat',
  SYSTEM_STATUS = 'system_status',
}

// 模态类型枚举
export enum ModalityType {
  TEXT = 'text',
  AUDIO = 'audio',
  VIDEO = 'video',
  IMAGE = 'image',
  PHYSIOLOGICAL = 'physiological',
}

// 情感类型枚举
export enum EmotionType {
  HAPPINESS = 'happiness',
  SADNESS = 'sadness',
  ANGER = 'anger',
  FEAR = 'fear',
  SURPRISE = 'surprise',
  DISGUST = 'disgust',
  NEUTRAL = 'neutral',
}

// 连接状态枚举
export enum ConnectionState {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  RECONNECTING = 'reconnecting',
  ERROR = 'error',
}

// WebSocket消息接口
export interface WebSocketMessage {
  type: MessageType
  data: Record<string, any>
  timestamp: string
  message_id?: string
  user_id?: string
}

// 情感输入消息接口
export interface EmotionInputMessage {
  text?: string
  audio_data?: string
  video_data?: string
  image_data?: string
  physiological_data?: Record<string, any>
  modalities: ModalityType[]
  timestamp: string
}

// 情感状态接口
export interface EmotionState {
  emotion: EmotionType
  intensity: number
  valence: number
  arousal: number
  dominance: number
  confidence: number
  timestamp?: string
}

// 多模态情感结果接口
export interface MultiModalEmotion {
  fused_emotion: EmotionState
  emotions: Record<string, EmotionState>
  confidence: number
  processing_time: number
}

// 个性特征接口
export interface PersonalityProfile {
  openness: number
  conscientiousness: number
  extraversion: number
  agreeableness: number
  neuroticism: number
  updated_at: string
}

// 共情响应接口
export interface EmpathyResponse {
  message: string
  response_type: string
  confidence: number
  generation_strategy: string
}

// 统一情感数据接口
export interface UnifiedEmotionalData {
  user_id: string
  timestamp: string
  recognition_result?: MultiModalEmotion
  emotional_state?: EmotionState
  personality_profile?: PersonalityProfile
  empathy_response?: EmpathyResponse
  confidence: number
  processing_time: number
  data_quality: number
}

// WebSocket配置接口
export interface WebSocketConfig {
  url: string
  userId: string
  autoReconnect?: boolean
  reconnectInterval?: number
  maxReconnectAttempts?: number
  heartbeatInterval?: number
}

// 连接统计接口
export interface ConnectionStats {
  total_connections: number
  active_connections: number
  messages_sent: number
  messages_received: number
  errors: number
  user_sessions: Record<
    string,
    {
      connected_at: string
      last_activity: string
      message_count: number
    }
  >
}

// 事件类型
export interface EmotionWebSocketEvents {
  connect: () => void
  disconnect: (reason?: string) => void
  emotion_result: (data: UnifiedEmotionalData) => void
  stream_status: (status: any) => void
  system_status: (status: any) => void
  error: (error: any) => void
  state_change: (state: ConnectionState) => void
  heartbeat: (data: any) => void
  message: (message: WebSocketMessage) => void
}

/**
 * 情感智能WebSocket客户端
 */
export class EmotionWebSocketService extends EventEmitter {
  private ws: WebSocket | null = null
  private config: WebSocketConfig
  private state: ConnectionState = ConnectionState.DISCONNECTED
  private reconnectAttempts = 0
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private heartbeatTimer: ReturnType<typeof setTimeout> | null = null
  private messageQueue: WebSocketMessage[] = []
  private stats = {
    messagesSent: 0,
    messagesReceived: 0,
    errors: 0,
    connectionTime: null as Date | null,
    lastHeartbeat: null as Date | null,
  }

  constructor(config: WebSocketConfig) {
    super()
    this.config = {
      autoReconnect: true,
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      ...config,
    }
  }

  /**
   * 连接到WebSocket服务器
   */
  async connect(): Promise<void> {
    if (
      this.state === ConnectionState.CONNECTING ||
      this.state === ConnectionState.CONNECTED
    ) {
      return
    }

    this.setState(ConnectionState.CONNECTING)

    try {
      const wsUrl = `${this.config.url}/emotion/${this.config.userId}`
      this.ws = new WebSocket(wsUrl)

      this.ws.onopen = this.handleOpen.bind(this)
      this.ws.onmessage = this.handleMessage.bind(this)
      this.ws.onclose = this.handleClose.bind(this)
      this.ws.onerror = this.handleError.bind(this)
    } catch (error) {
      this.setState(ConnectionState.ERROR)
      this.emit('error', error)
    }
  }

  /**
   * 断开WebSocket连接
   */
  disconnect(): void {
    this.config.autoReconnect = false
    this.clearTimers()

    if (this.ws) {
      this.ws.close()
      this.ws = null
    }

    this.setState(ConnectionState.DISCONNECTED)
  }

  /**
   * 发送情感输入数据
   */
  async sendEmotionInput(input: EmotionInputMessage): Promise<void> {
    const message: WebSocketMessage = {
      type: MessageType.EMOTION_INPUT,
      data: input,
      timestamp: new Date().toISOString(),
      message_id: this.generateMessageId(),
      user_id: this.config.userId,
    }

    await this.sendMessage(message)
  }

  /**
   * 发送文本情感数据
   */
  async sendTextEmotion(text: string): Promise<void> {
    await this.sendEmotionInput({
      text,
      modalities: [ModalityType.TEXT],
      timestamp: new Date().toISOString(),
    })
  }

  /**
   * 发送音频情感数据
   */
  async sendAudioEmotion(audioBlob: Blob): Promise<void> {
    const audioBase64 = await this.blobToBase64(audioBlob)
    await this.sendEmotionInput({
      audio_data: audioBase64,
      modalities: [ModalityType.AUDIO],
      timestamp: new Date().toISOString(),
    })
  }

  /**
   * 发送视频情感数据
   */
  async sendVideoEmotion(videoBlob: Blob): Promise<void> {
    const videoBase64 = await this.blobToBase64(videoBlob)
    await this.sendEmotionInput({
      video_data: videoBase64,
      modalities: [ModalityType.VIDEO],
      timestamp: new Date().toISOString(),
    })
  }

  /**
   * 发送图像情感数据
   */
  async sendImageEmotion(imageFile: File): Promise<void> {
    const imageBase64 = await this.fileToBase64(imageFile)
    await this.sendEmotionInput({
      image_data: imageBase64,
      modalities: [ModalityType.IMAGE],
      timestamp: new Date().toISOString(),
    })
  }

  /**
   * 发送生理数据
   */
  async sendPhysiologicalData(data: Record<string, any>): Promise<void> {
    await this.sendEmotionInput({
      physiological_data: data,
      modalities: [ModalityType.PHYSIOLOGICAL],
      timestamp: new Date().toISOString(),
    })
  }

  /**
   * 请求流状态
   */
  async requestStreamStatus(): Promise<void> {
    const message: WebSocketMessage = {
      type: MessageType.STREAM_STATUS,
      data: { request: 'status' },
      timestamp: new Date().toISOString(),
      message_id: this.generateMessageId(),
      user_id: this.config.userId,
    }

    await this.sendMessage(message)
  }

  /**
   * 发送心跳
   */
  private async sendHeartbeat(): Promise<void> {
    const message: WebSocketMessage = {
      type: MessageType.HEARTBEAT,
      data: {
        client_time: new Date().toISOString(),
        stats: this.getConnectionStats(),
      },
      timestamp: new Date().toISOString(),
      message_id: this.generateMessageId(),
      user_id: this.config.userId,
    }

    await this.sendMessage(message)
  }

  /**
   * 发送消息
   */
  private async sendMessage(message: WebSocketMessage): Promise<void> {
    if (this.state !== ConnectionState.CONNECTED) {
      this.messageQueue.push(message)
      return
    }

    try {
      const messageStr = JSON.stringify(message)
      this.ws?.send(messageStr)
      this.stats.messagesSent++
      this.emit('message', message)
    } catch (error) {
      this.stats.errors++
      this.emit('error', error)
    }
  }

  /**
   * 处理连接打开
   */
  private handleOpen(): void {
    this.setState(ConnectionState.CONNECTED)
    this.stats.connectionTime = new Date()
    this.reconnectAttempts = 0

    // 发送队列中的消息
    this.flushMessageQueue()

    // 启动心跳
    this.startHeartbeat()

    this.emit('connect')
  }

  /**
   * 处理收到的消息
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data)
      this.stats.messagesReceived++
      this.stats.lastHeartbeat = new Date()

      switch (message.type) {
        case MessageType.CONNECT:
          // 连接确认消息
          break

        case MessageType.EMOTION_RESULT:
          this.emit('emotion_result', message.data as UnifiedEmotionalData)
          break

        case MessageType.STREAM_STATUS:
          this.emit('stream_status', message.data)
          break

        case MessageType.SYSTEM_STATUS:
          this.emit('system_status', message.data)
          break

        case MessageType.ERROR:
          this.stats.errors++
          this.emit('error', message.data)
          break

        case MessageType.HEARTBEAT:
          this.emit('heartbeat', message.data)
          break

        default:
          logger.warn('未知消息类型:', message.type)
      }

      this.emit('message', message)
    } catch (error) {
      this.stats.errors++
      this.emit('error', error)
    }
  }

  /**
   * 处理连接关闭
   */
  private handleClose(event: CloseEvent): void {
    this.clearTimers()

    const reason = this.getCloseReason(event.code)
    this.emit('disconnect', reason)

    if (
      this.config.autoReconnect &&
      this.reconnectAttempts < this.config.maxReconnectAttempts!
    ) {
      this.setState(ConnectionState.RECONNECTING)
      this.scheduleReconnect()
    } else {
      this.setState(ConnectionState.DISCONNECTED)
    }
  }

  /**
   * 处理连接错误
   */
  private handleError(event: Event): void {
    this.stats.errors++
    this.setState(ConnectionState.ERROR)
    this.emit('error', event)
  }

  /**
   * 设置连接状态
   */
  private setState(newState: ConnectionState): void {
    if (this.state !== newState) {
      this.state = newState
      this.emit('state_change', newState)
    }
  }

  /**
   * 安排重连
   */
  private scheduleReconnect(): void {
    this.reconnectAttempts++

    const delay = Math.min(
      this.config.reconnectInterval! * Math.pow(2, this.reconnectAttempts - 1),
      30000 // 最大30秒
    )

    this.reconnectTimer = setTimeout(() => {
      this.connect()
    }, delay)
  }

  /**
   * 启动心跳
   */
  private startHeartbeat(): void {
    if (this.config.heartbeatInterval! > 0) {
      this.heartbeatTimer = setInterval(() => {
        this.sendHeartbeat()
      }, this.config.heartbeatInterval!)
    }
  }

  /**
   * 清理定时器
   */
  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  /**
   * 发送消息队列中的所有消息
   */
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift()
      if (message) {
        this.sendMessage(message)
      }
    }
  }

  /**
   * 生成消息ID
   */
  private generateMessageId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
  }

  /**
   * 获取关闭原因
   */
  private getCloseReason(code: number): string {
    switch (code) {
      case 1000:
        return 'Normal closure'
      case 1001:
        return 'Going away'
      case 1002:
        return 'Protocol error'
      case 1003:
        return 'Unsupported data'
      case 1006:
        return 'Connection lost'
      case 1011:
        return 'Server error'
      default:
        return `Unknown reason (${code})`
    }
  }

  /**
   * Blob转Base64
   */
  private blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        resolve(result.split(',')[1]) // 移除data:type/subtype;base64,前缀
      }
      reader.onerror = reject
      reader.readAsDataURL(blob)
    })
  }

  /**
   * File转Base64
   */
  private fileToBase64(file: File): Promise<string> {
    return this.blobToBase64(file)
  }

  /**
   * 获取连接状态
   */
  getConnectionState(): ConnectionState {
    return this.state
  }

  /**
   * 获取连接统计
   */
  getConnectionStats() {
    return {
      state: this.state,
      messagesSent: this.stats.messagesSent,
      messagesReceived: this.stats.messagesReceived,
      errors: this.stats.errors,
      connectionTime: this.stats.connectionTime?.toISOString(),
      lastHeartbeat: this.stats.lastHeartbeat?.toISOString(),
      reconnectAttempts: this.reconnectAttempts,
      queuedMessages: this.messageQueue.length,
    }
  }

  /**
   * 检查是否已连接
   */
  isConnected(): boolean {
    return this.state === ConnectionState.CONNECTED
  }

  /**
   * 更新配置
   */
  updateConfig(newConfig: Partial<WebSocketConfig>): void {
    this.config = { ...this.config, ...newConfig }
  }
}

// 工厂函数
export function createEmotionWebSocketService(
  config: WebSocketConfig
): EmotionWebSocketService {
  return new EmotionWebSocketService(config)
}

// 默认实例（可选）
export let defaultEmotionWebSocket: EmotionWebSocketService | null = null

export function initializeDefaultWebSocket(
  config: WebSocketConfig
): EmotionWebSocketService {
  defaultEmotionWebSocket = createEmotionWebSocketService(config)
  return defaultEmotionWebSocket
}

export function getDefaultWebSocket(): EmotionWebSocketService | null {
  return defaultEmotionWebSocket
}
