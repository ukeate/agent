import { buildWsUrl } from '../utils/apiBase'
import React, { useState, useEffect, useRef } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Typography,
  Input,
  Table,
  Tag,
  Alert,
  Statistic,
  message,
  Badge,
  Timeline,
  List,
} from 'antd'
import {
  WifiOutlined,
  DisconnectOutlined,
  SendOutlined,
  HeartOutlined,
  MessageOutlined,
  UserOutlined,
  ClockCircleOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import dayjs from 'dayjs'
import {
  webSocketConnectionService,
  type WebSocketConnectionsResponse,
  type WebSocketConnection,
} from '../services/webSocketConnectionService'

const { Title, Text } = Typography
const { TextArea } = Input

interface WebSocketMessage {
  id: string
  type: 'sent' | 'received'
  content: any
  timestamp: string
  messageType?: string
}

// 使用服务中定义的类型，保留本地的消息类型
type ConnectionInfo = WebSocketConnection
type ConnectionStatus = WebSocketConnectionsResponse

const WebSocketManagementPage: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false)
  const [messages, setMessages] = useState<WebSocketMessage[]>([])
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus | null>(null)
  const [customMessage, setCustomMessage] = useState('')
  const [autoReconnect, setAutoReconnect] = useState(true)
  const [messagesSent, setMessagesSent] = useState(0)
  const [messagesReceived, setMessagesReceived] = useState(0)
  const [connectionTime, setConnectionTime] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const heartbeatIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  )
  const statusCheckIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  )

  useEffect(() => {
    // 组件加载时获取连接状态
    fetchConnectionStatus()

    // 定期检查连接状态
    statusCheckIntervalRef.current = setInterval(fetchConnectionStatus, 5000)

    return () => {
      disconnect()
      if (statusCheckIntervalRef.current) {
        clearInterval(statusCheckIntervalRef.current)
      }
    }
  }, [])

  const fetchConnectionStatus = async () => {
    try {
      const data = await webSocketConnectionService.getConnections()
      setConnectionStatus(data)
    } catch (error) {
      logger.error('获取连接状态失败:', error)
      message.error('获取WebSocket连接状态失败')
    }
  }

  const connect = () => {
    if (isConnected) {
      message.warning('WebSocket已连接')
      return
    }

    try {
      const wsUrl = buildWsUrl('/ws')
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        setIsConnected(true)
        setConnectionTime(dayjs().format('YYYY-MM-DD HH:mm:ss'))
        message.success('WebSocket连接成功')

        // 开始心跳检测
        startHeartbeat()

        // 发送连接消息
        const connectMessage = {
          type: 'connection',
          user_id: 'admin',
          timestamp: new Date().toISOString(),
        }
        sendMessage(connectMessage)
      }

      wsRef.current.onmessage = event => {
        try {
          const data = JSON.parse(event.data)
          handleReceivedMessage(data)
        } catch (error) {
          // 处理非JSON消息
          handleReceivedMessage({
            type: 'raw_message',
            data: event.data,
            timestamp: new Date().toISOString(),
          })
        }
      }

      wsRef.current.onclose = event => {
        setIsConnected(false)
        setConnectionTime(null)
        stopHeartbeat()
        wsRef.current = null

        message.info(
          `WebSocket连接关闭 (${event.code}: ${event.reason || '正常关闭'})`
        )

        // 自动重连
        if (autoReconnect && event.code !== 1000) {
          reconnectTimeoutRef.current = setTimeout(() => {
            message.info('尝试自动重连...')
            connect()
          }, 3000)
        }
      }

      wsRef.current.onerror = error => {
        message.error('WebSocket连接错误')
        logger.error('WebSocket错误:', error)
        if (
          wsRef.current &&
          wsRef.current.readyState !== WebSocket.CLOSING &&
          wsRef.current.readyState !== WebSocket.CLOSED
        ) {
          wsRef.current.close()
        }
      }
    } catch (error) {
      message.error('无法创建WebSocket连接')
      logger.error('连接错误:', error)
    }
  }

  const disconnect = () => {
    if (wsRef.current) {
      wsRef.current.close(1000, '用户主动断开')
      wsRef.current = null
    }

    stopHeartbeat()

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
  }

  const startHeartbeat = () => {
    heartbeatIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const pingMessage = {
          type: 'ping',
          timestamp: new Date().toISOString(),
        }
        sendMessage(pingMessage)
      }
    }, 30000) // 每30秒发送心跳
  }

  const stopHeartbeat = () => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current)
      heartbeatIntervalRef.current = null
    }
  }

  const sendMessage = (content: any, messageType?: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      message.warning('WebSocket未连接')
      return
    }

    try {
      const messageStr =
        typeof content === 'string' ? content : JSON.stringify(content)
      wsRef.current.send(messageStr)

      // 记录发送的消息
      const newMessage: WebSocketMessage = {
        id: Date.now().toString(),
        type: 'sent',
        content,
        timestamp: dayjs().format('YYYY-MM-DD HH:mm:ss'),
        messageType:
          messageType ||
          (typeof content === 'object' ? content.type : 'custom'),
      }

      setMessages(prev => [newMessage, ...prev].slice(0, 100)) // 保留最新100条
      setMessagesSent(prev => prev + 1)
    } catch (error) {
      message.error('发送消息失败')
      logger.error('发送消息错误:', error)
    }
  }

  const handleReceivedMessage = (data: any) => {
    const newMessage: WebSocketMessage = {
      id: Date.now().toString(),
      type: 'received',
      content: data,
      timestamp: dayjs().format('YYYY-MM-DD HH:mm:ss'),
      messageType: data.type || 'response',
    }

    setMessages(prev => [newMessage, ...prev].slice(0, 100))
    setMessagesReceived(prev => prev + 1)
  }

  const sendCustomMessage = () => {
    if (!customMessage.trim()) {
      message.warning('请输入消息内容')
      return
    }

    try {
      const messageData = JSON.parse(customMessage)
      sendMessage(messageData, 'custom')
    } catch (error) {
      // 如果不是JSON，作为普通文本发送
      sendMessage(customMessage, 'text')
    }

    setCustomMessage('')
  }

  const sendTestMessage = () => {
    const testMessage = {
      type: 'test',
      message: '这是一个测试消息',
      user_id: 'admin',
      timestamp: new Date().toISOString(),
    }
    sendMessage(testMessage, 'test')
  }

  const clearMessages = () => {
    setMessages([])
    setMessagesSent(0)
    setMessagesReceived(0)
  }

  // 新增连接管理功能
  const disconnectSpecificConnection = async (connectionIndex: number) => {
    try {
      const result =
        await webSocketConnectionService.disconnectConnection(connectionIndex)
      if (result.success) {
        message.success(`连接 ${connectionIndex} 已断开`)
        fetchConnectionStatus() // 刷新状态
      } else {
        message.error(result.message)
      }
    } catch (error) {
      message.error('断开连接失败')
      logger.error('断开连接失败:', error)
    }
  }

  const disconnectAllConnections = async () => {
    try {
      const result = await webSocketConnectionService.disconnectAllConnections()
      if (result.success) {
        message.success(`已断开 ${result.disconnected_count} 个连接`)
        fetchConnectionStatus() // 刷新状态
      } else {
        message.error(result.message)
      }
    } catch (error) {
      message.error('断开所有连接失败')
      logger.error('断开所有连接失败:', error)
    }
  }

  const broadcastMessage = async () => {
    if (!customMessage.trim()) {
      message.warning('请输入要广播的消息内容')
      return
    }

    try {
      let messageData: any
      try {
        messageData = JSON.parse(customMessage)
      } catch {
        messageData = { type: 'broadcast', content: customMessage }
      }

      const result =
        await webSocketConnectionService.broadcastMessage(messageData)
      if (result.success) {
        message.success(`消息已广播到 ${result.sent_count} 个连接`)
        setCustomMessage('')
      } else {
        message.error(result.message)
      }
    } catch (error) {
      message.error('广播消息失败')
      logger.error('广播消息失败:', error)
    }
  }

  const sendToSpecificConnection = async (connectionIndex: number) => {
    if (!customMessage.trim()) {
      message.warning('请输入要发送的消息内容')
      return
    }

    try {
      let messageData: any
      try {
        messageData = JSON.parse(customMessage)
      } catch {
        messageData = { type: 'direct', content: customMessage }
      }

      const result = await webSocketConnectionService.sendMessageToConnection(
        connectionIndex,
        messageData
      )
      if (result.success) {
        message.success(`消息已发送到连接 ${connectionIndex}`)
        setCustomMessage('')
      } else {
        message.error(result.message)
      }
    } catch (error) {
      message.error('发送消息失败')
      logger.error('发送消息失败:', error)
    }
  }

  const messageColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 180,
    },
    {
      title: '方向',
      dataIndex: 'type',
      key: 'type',
      width: 80,
      render: (type: string) => (
        <Tag color={type === 'sent' ? 'blue' : 'green'}>
          {type === 'sent' ? '发送' : '接收'}
        </Tag>
      ),
    },
    {
      title: '类型',
      dataIndex: 'messageType',
      key: 'messageType',
      width: 100,
      render: (type: string) => <Tag>{type}</Tag>,
    },
    {
      title: '内容',
      dataIndex: 'content',
      key: 'content',
      render: (content: any) => (
        <pre
          style={{
            maxWidth: 400,
            overflow: 'auto',
            fontSize: '12px',
            margin: 0,
            whiteSpace: 'pre-wrap',
          }}
        >
          {typeof content === 'string'
            ? content
            : JSON.stringify(content, null, 2)}
        </pre>
      ),
    },
  ]

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <ApiOutlined /> WebSocket实时通信管理
      </Title>
      <Text type="secondary">
        管理和监控WebSocket实时通信连接，测试消息传输功能
      </Text>

      {/* 连接状态面板 */}
      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="连接状态"
              value={isConnected ? '已连接' : '未连接'}
              prefix={
                isConnected ? <CheckCircleOutlined /> : <WarningOutlined />
              }
              valueStyle={{ color: isConnected ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="连接时间"
              value={connectionTime || '--'}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="发送消息数"
              value={messagesSent}
              prefix={<SendOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="接收消息数"
              value={messagesReceived}
              prefix={<MessageOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 控制面板 */}
      <Card title="连接控制" style={{ marginTop: 24 }}>
        <Space size="large">
          {!isConnected ? (
            <Button type="primary" icon={<WifiOutlined />} onClick={connect}>
              连接WebSocket
            </Button>
          ) : (
            <Button danger icon={<DisconnectOutlined />} onClick={disconnect}>
              断开连接
            </Button>
          )}

          <Button
            icon={<HeartOutlined />}
            onClick={() => sendMessage({ type: 'ping' })}
            disabled={!isConnected}
          >
            发送心跳
          </Button>

          <Button onClick={sendTestMessage} disabled={!isConnected}>
            发送测试消息
          </Button>

          <Button icon={<ReloadOutlined />} onClick={fetchConnectionStatus}>
            刷新状态
          </Button>

          <Button onClick={clearMessages}>清空消息</Button>
        </Space>
      </Card>

      {/* 服务器连接状态 */}
      {connectionStatus && (
        <Card title="服务器连接状态" style={{ marginTop: 24 }}>
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Badge
                status="success"
                text={`活跃连接数: ${connectionStatus.active_connections}`}
              />
            </Col>
            <Col span={12}>
              <Text type="secondary">
                最后更新: {dayjs().format('YYYY-MM-DD HH:mm:ss')}
              </Text>
            </Col>
          </Row>

          {connectionStatus.connection_details.length > 0 && (
            <List
              size="small"
              header={<div>连接详情</div>}
              bordered
              dataSource={connectionStatus.connection_details}
              renderItem={item => (
                <List.Item
                  actions={[
                    <Button
                      key="send"
                      size="small"
                      icon={<SendOutlined />}
                      onClick={() => sendToSpecificConnection(item.index)}
                      disabled={!customMessage.trim()}
                      title="发送消息到此连接"
                    >
                      发送
                    </Button>,
                    <Button
                      key="disconnect"
                      size="small"
                      danger
                      icon={<DisconnectOutlined />}
                      onClick={() => disconnectSpecificConnection(item.index)}
                      title="断开此连接"
                    >
                      断开
                    </Button>,
                  ]}
                >
                  <List.Item.Meta
                    avatar={<Badge status="processing" />}
                    title={<Text strong>连接#{item.index}</Text>}
                    description={
                      <Space direction="vertical" size="small">
                        <div>
                          <Tag
                            color={item.state === 'connected' ? 'green' : 'red'}
                          >
                            {item.state}
                          </Tag>
                          <Text type="secondary">
                            连接时间:{' '}
                            {dayjs(item.connected_at).format('MM-DD HH:mm:ss')}
                          </Text>
                        </div>
                        {item.client_info && (
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            会话ID: {item.client_info.session_id || 'N/A'}
                          </Text>
                        )}
                        {item.messages_sent !== undefined && (
                          <Space size="large">
                            <Text type="secondary">
                              发送: {item.messages_sent || 0}
                            </Text>
                            <Text type="secondary">
                              接收: {item.messages_received || 0}
                            </Text>
                          </Space>
                        )}
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          )}
        </Card>
      )}

      {/* 连接管理和消息发送 */}
      <Card title="连接管理与消息发送" style={{ marginTop: 24 }}>
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          {/* 全局连接管理按钮 */}
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Button
                danger
                icon={<DisconnectOutlined />}
                onClick={disconnectAllConnections}
                disabled={
                  !connectionStatus || connectionStatus.active_connections === 0
                }
                block
              >
                断开所有连接 ({connectionStatus?.active_connections || 0})
              </Button>
            </Col>
            <Col span={12}>
              <Button
                icon={<ReloadOutlined />}
                onClick={fetchConnectionStatus}
                block
              >
                刷新连接状态
              </Button>
            </Col>
          </Row>

          {/* 消息发送区域 */}
          <div>
            <Text strong style={{ marginBottom: 8, display: 'block' }}>
              消息内容:
            </Text>
            <TextArea
              value={customMessage}
              onChange={e => setCustomMessage(e.target.value)}
              name="ws-custom-message"
              placeholder='输入JSON消息，如: {"type": "custom", "message": "hello"}，或输入普通文本'
              rows={3}
              style={{ marginBottom: 12 }}
            />
            <Space wrap>
              <Button
                type="primary"
                icon={<SendOutlined />}
                onClick={sendCustomMessage}
                disabled={!isConnected || !customMessage.trim()}
              >
                发送到当前连接
              </Button>
              <Button
                icon={<ApiOutlined />}
                onClick={broadcastMessage}
                disabled={
                  !customMessage.trim() ||
                  !connectionStatus ||
                  connectionStatus.active_connections === 0
                }
              >
                广播到所有连接
              </Button>
              <Button
                icon={<HeartOutlined />}
                onClick={sendTestMessage}
                disabled={!isConnected}
              >
                发送测试消息
              </Button>
            </Space>
          </div>
        </Space>
      </Card>

      {/* 消息历史 */}
      <Card title="消息历史" style={{ marginTop: 24 }}>
        <Table
          columns={messageColumns}
          dataSource={messages}
          rowKey="id"
          pagination={{ pageSize: 20, showSizeChanger: true }}
          scroll={{ y: 400 }}
          size="small"
        />
      </Card>
    </div>
  )
}

export default WebSocketManagementPage
