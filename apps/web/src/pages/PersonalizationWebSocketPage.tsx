import React, { useState, useEffect, useRef } from 'react'
import { Card, Row, Col, Button, Space, Typography, Input, Select, message, Badge, Table, Tag, Alert, Progress, Statistic } from 'antd'
import { 
  WifiOutlined,
  DisconnectOutlined,
  SendOutlined,
  HeartOutlined,
  ApiOutlined,
  MessageOutlined,
  UserOutlined,
  RocketOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  DashboardOutlined,
  LineChartOutlined,
  ClockCircleOutlined
} from '@ant-design/icons'
import { Line } from '@ant-design/plots'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { Option } = Select

interface WebSocketMessage {
  id: string
  type: 'sent' | 'received'
  messageType: string
  content: any
  timestamp: string
}

interface RecommendationResult {
  request_id: string
  user_id: string
  recommendations: Array<{
    item_id: string
    score: number
    confidence: number
    explanation: string
  }>
  latency_ms: number
  model_version: string
  explanation: string
}

interface ConnectionStats {
  connected: boolean
  connectionTime?: string
  messagesSeent: number
  messagesReceived: number
  lastPingTime?: string
  avgLatency: number
}

const PersonalizationWebSocketPage: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false)
  const [userId, setUserId] = useState('demo_user_123')
  const [scenario, setScenario] = useState('content_discovery')
  const [nRecommendations, setNRecommendations] = useState(5)
  const [contextData, setContextData] = useState('{"page": "homepage", "device": "mobile"}')
  const [messages, setMessages] = useState<WebSocketMessage[]>([])
  const [connectionStats, setConnectionStats] = useState<ConnectionStats>({
    connected: false,
    messagesSeent: 0,
    messagesReceived: 0,
    avgLatency: 0
  })
  const [latencyHistory, setLatencyHistory] = useState<any[]>([])
  const [autoHeartbeat, setAutoHeartbeat] = useState(false)
  
  const wsRef = useRef<WebSocket | null>(null)
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const pingTimeRef = useRef<number>(0)

  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [])

  const connect = () => {
    if (isConnected) {
      message.warning('WebSocket已连接')
      return
    }

    try {
      // 实际项目中这里应该是真实的WebSocket URL
      const wsUrl = `ws://localhost:8000/api/v1/personalization/stream`
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        setIsConnected(true)
        setConnectionStats(prev => ({
          ...prev,
          connected: true,
          connectionTime: new Date().toLocaleTimeString()
        }))
        
        message.success('WebSocket连接成功')
        
        // 发送认证消息
        sendAuthMessage()
        
        // 开始自动心跳
        if (autoHeartbeat) {
          startHeartbeat()
        }
      }

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data)
        handleReceivedMessage(data)
      }

      wsRef.current.onclose = () => {
        setIsConnected(false)
        setConnectionStats(prev => ({
          ...prev,
          connected: false
        }))
        stopHeartbeat()
        message.info('WebSocket连接已关闭')
      }

      wsRef.current.onerror = (error) => {
        message.error('WebSocket连接错误')
        console.error('WebSocket error:', error)
      }

    } catch (error) {
      message.error('连接失败，将使用模拟模式')
      // 启用模拟模式
      enableMockMode()
    }
  }

  const disconnect = () => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    stopHeartbeat()
  }

  const enableMockMode = () => {
    // 模拟连接成功
    setTimeout(() => {
      setIsConnected(true)
      setConnectionStats(prev => ({
        ...prev,
        connected: true,
        connectionTime: new Date().toLocaleTimeString()
      }))
      message.success('已连接到模拟WebSocket服务')
      
      // 模拟欢迎消息
      const welcomeMessage = {
        type: 'connected',
        user_id: userId,
        timestamp: new Date().toISOString()
      }
      handleReceivedMessage(welcomeMessage)
    }, 500)
  }

  const sendAuthMessage = () => {
    const authMessage = {
      user_id: userId
    }
    sendMessage(authMessage, 'auth')
  }

  const sendMessage = (content: any, messageType: string) => {
    const messageId = Date.now().toString()
    const timestamp = new Date().toLocaleTimeString()
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(content))
    }

    // 添加到消息历史
    const newMessage: WebSocketMessage = {
      id: messageId,
      type: 'sent',
      messageType,
      content,
      timestamp
    }
    
    setMessages(prev => [...prev, newMessage])
    setConnectionStats(prev => ({
      ...prev,
      messagesSeent: prev.messagesSeent + 1
    }))

    // 模拟模式下的响应
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setTimeout(() => {
        simulateResponse(content, messageType)
      }, 200 + Math.random() * 300)
    }
  }

  const simulateResponse = (originalContent: any, originalType: string) => {
    let response: any = {}

    switch (originalType) {
      case 'auth':
        response = {
          type: 'connected',
          user_id: originalContent.user_id,
          timestamp: new Date().toISOString()
        }
        break
      
      case 'recommendation':
        response = {
          type: 'recommendations',
          data: {
            request_id: `req_${Date.now()}`,
            user_id: userId,
            recommendations: [
              {
                item_id: `item_${Math.floor(Math.random() * 1000)}`,
                score: 0.9 + Math.random() * 0.1,
                confidence: 0.8 + Math.random() * 0.2,
                explanation: '基于用户历史行为推荐'
              },
              {
                item_id: `item_${Math.floor(Math.random() * 1000)}`,
                score: 0.8 + Math.random() * 0.1,
                confidence: 0.75 + Math.random() * 0.15,
                explanation: '相似用户喜欢的内容'
              },
              {
                item_id: `item_${Math.floor(Math.random() * 1000)}`,
                score: 0.7 + Math.random() * 0.1,
                confidence: 0.7 + Math.random() * 0.1,
                explanation: '热门内容推荐'
              }
            ].slice(0, nRecommendations),
            latency_ms: 30 + Math.random() * 50,
            model_version: 'v1.2.3',
            explanation: '基于协同过滤和内容匹配的混合推荐'
          },
          timestamp: new Date().toISOString()
        }
        break
      
      case 'feedback':
        response = {
          type: 'feedback_received',
          timestamp: new Date().toISOString()
        }
        break
      
      case 'ping':
        response = {
          type: 'pong',
          timestamp: new Date().toISOString()
        }
        break
      
      default:
        response = {
          type: 'error',
          message: `未知消息类型: ${originalType}`
        }
    }

    handleReceivedMessage(response)
  }

  const handleReceivedMessage = (data: any) => {
    const messageId = Date.now().toString()
    const timestamp = new Date().toLocaleTimeString()
    
    const newMessage: WebSocketMessage = {
      id: messageId,
      type: 'received',
      messageType: data.type,
      content: data,
      timestamp
    }
    
    setMessages(prev => [...prev, newMessage])
    setConnectionStats(prev => ({
      ...prev,
      messagesReceived: prev.messagesReceived + 1
    }))

    // 处理pong消息，计算延迟
    if (data.type === 'pong' && pingTimeRef.current > 0) {
      const latency = Date.now() - pingTimeRef.current
      setLatencyHistory(prev => {
        const newHistory = [...prev, {
          timestamp: new Date().toLocaleTimeString(),
          latency
        }].slice(-20) // 保留最近20个数据点
        
        // 更新平均延迟
        const avgLatency = newHistory.reduce((sum, item) => sum + item.latency, 0) / newHistory.length
        setConnectionStats(prevStats => ({
          ...prevStats,
          avgLatency: Math.round(avgLatency),
          lastPingTime: new Date().toLocaleTimeString()
        }))
        
        return newHistory
      })
      pingTimeRef.current = 0
    }
  }

  const sendRecommendationRequest = () => {
    try {
      const context = JSON.parse(contextData)
      const requestMessage = {
        type: 'request',
        context,
        n_recommendations: nRecommendations,
        scenario
      }
      sendMessage(requestMessage, 'recommendation')
    } catch (error) {
      message.error('上下文数据JSON格式错误')
    }
  }

  const sendFeedback = (itemId: string, feedbackType: string) => {
    const feedbackMessage = {
      type: 'feedback',
      item_id: itemId,
      feedback_type: feedbackType,
      feedback_value: 1.0,
      context: { source: 'websocket_test' }
    }
    sendMessage(feedbackMessage, 'feedback')
  }

  const sendPing = () => {
    pingTimeRef.current = Date.now()
    const pingMessage = { type: 'ping' }
    sendMessage(pingMessage, 'ping')
  }

  const startHeartbeat = () => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current)
    }
    
    heartbeatIntervalRef.current = setInterval(() => {
      if (isConnected) {
        sendPing()
      }
    }, 30000) // 每30秒发送一次心跳
  }

  const stopHeartbeat = () => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current)
      heartbeatIntervalRef.current = null
    }
  }

  const clearMessages = () => {
    setMessages([])
    setConnectionStats(prev => ({
      ...prev,
      messagesSeent: 0,
      messagesReceived: 0
    }))
  }

  const messageColumns: ColumnsType<WebSocketMessage> = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 100
    },
    {
      title: '方向',
      dataIndex: 'type',
      key: 'type',
      width: 80,
      render: (type) => (
        <Tag color={type === 'sent' ? 'blue' : 'green'}>
          {type === 'sent' ? '发送' : '接收'}
        </Tag>
      )
    },
    {
      title: '类型',
      dataIndex: 'messageType',
      key: 'messageType',
      width: 120,
      render: (type) => <Tag>{type}</Tag>
    },
    {
      title: '内容',
      dataIndex: 'content',
      key: 'content',
      render: (content) => (
        <pre style={{ 
          maxWidth: 400, 
          overflow: 'auto', 
          fontSize: '12px',
          margin: 0,
          whiteSpace: 'pre-wrap'
        }}>
          {JSON.stringify(content, null, 2)}
        </pre>
      )
    }
  ]

  // 延迟趋势图配置
  const latencyTrendConfig = {
    data: latencyHistory,
    xField: 'timestamp',
    yField: 'latency',
    smooth: true,
    color: '#1890ff',
    point: { size: 3 },
    yAxis: {
      title: { text: '延迟 (ms)' }
    },
    height: 200
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <WifiOutlined /> WebSocket实时推荐
      </Title>
      <Paragraph type="secondary">
        测试和管理个性化引擎的WebSocket实时推荐功能
      </Paragraph>

      {/* 连接控制 */}
      <Card title="连接控制" style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>用户ID:</Text>
              <Input
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                disabled={isConnected}
                placeholder="输入用户ID"
              />
            </Space>
          </Col>
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>连接状态:</Text>
              <Space>
                <Badge 
                  status={isConnected ? 'success' : 'error'} 
                  text={isConnected ? '已连接' : '未连接'} 
                />
                {isConnected ? (
                  <Button 
                    danger 
                    icon={<DisconnectOutlined />}
                    onClick={disconnect}
                  >
                    断开
                  </Button>
                ) : (
                  <Button 
                    type="primary" 
                    icon={<WifiOutlined />}
                    onClick={connect}
                  >
                    连接
                  </Button>
                )}
              </Space>
            </Space>
          </Col>
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>心跳检测:</Text>
              <Space>
                <Button
                  icon={<HeartOutlined />}
                  onClick={sendPing}
                  disabled={!isConnected}
                >
                  Ping
                </Button>
                <Button
                  type={autoHeartbeat ? 'primary' : 'default'}
                  onClick={() => {
                    setAutoHeartbeat(!autoHeartbeat)
                    if (!autoHeartbeat && isConnected) {
                      startHeartbeat()
                    } else {
                      stopHeartbeat()
                    }
                  }}
                >
                  自动心跳
                </Button>
              </Space>
            </Space>
          </Col>
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>消息控制:</Text>
              <Button onClick={clearMessages}>清空消息</Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 连接统计 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="连接时间"
              value={connectionStats.connectionTime || '--'}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="发送消息"
              value={connectionStats.messagesSeent}
              prefix={<SendOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="接收消息"
              value={connectionStats.messagesReceived}
              prefix={<MessageOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均延迟"
              value={connectionStats.avgLatency}
              suffix="ms"
              prefix={<ThunderboltOutlined />}
              valueStyle={{ 
                color: connectionStats.avgLatency > 100 ? '#ff4d4f' : '#52c41a' 
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* 推荐请求 */}
      <Card title="推荐请求" style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]}>
          <Col span={8}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>推荐场景:</Text>
              <Select
                value={scenario}
                onChange={setScenario}
                style={{ width: '100%' }}
              >
                <Option value="content_discovery">内容发现</Option>
                <Option value="search_results">搜索结果</Option>
                <Option value="product_recommendations">商品推荐</Option>
                <Option value="personalized">个性化推荐</Option>
              </Select>
            </Space>
          </Col>
          <Col span={8}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>推荐数量:</Text>
              <Select
                value={nRecommendations}
                onChange={setNRecommendations}
                style={{ width: '100%' }}
              >
                <Option value={3}>3个</Option>
                <Option value={5}>5个</Option>
                <Option value={10}>10个</Option>
                <Option value={20}>20个</Option>
              </Select>
            </Space>
          </Col>
          <Col span={8}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>操作:</Text>
              <Button
                type="primary"
                icon={<RocketOutlined />}
                onClick={sendRecommendationRequest}
                disabled={!isConnected}
                block
              >
                获取推荐
              </Button>
            </Space>
          </Col>
        </Row>
        <Row style={{ marginTop: 16 }}>
          <Col span={24}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>上下文数据 (JSON):</Text>
              <TextArea
                value={contextData}
                onChange={(e) => setContextData(e.target.value)}
                rows={3}
                placeholder='{"page": "homepage", "device": "mobile"}'
              />
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 延迟监控 */}
      {latencyHistory.length > 0 && (
        <Card title="延迟监控" style={{ marginBottom: 24 }}>
          <Line {...latencyTrendConfig} />
        </Card>
      )}

      {/* 消息历史 */}
      <Card title="消息历史">
        <Table
          columns={messageColumns}
          dataSource={messages}
          rowKey="id"
          scroll={{ y: 400 }}
          size="small"
          pagination={false}
        />
      </Card>
    </div>
  )
}

export default PersonalizationWebSocketPage