import React, { useState, useEffect, useRef } from 'react';
import { Card, Button, Alert, Tabs, Row, Col, Statistic, Table, Space, Tag, message, Typography } from 'antd';
import { PlayCircleOutlined, StopOutlined, ThunderboltOutlined, MonitorOutlined, ApiOutlined } from '@ant-design/icons';
import { 
import { logger } from '../utils/logger'
  streamingService,
  StreamingMetrics,
  BackpressureStatus,
  QueueStatus,
  SessionMetrics
} from '../services/streamingService';

const { Title } = Typography;

const StreamingMonitorPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // 系统指标状态
  const [systemMetrics, setSystemMetrics] = useState<StreamingMetrics | null>(null);
  const [backpressureStatus, setBackpressureStatus] = useState<BackpressureStatus | null>(null);
  const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null);
  const [sessions, setSessions] = useState<SessionMetrics[]>([]);
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [activeSessionId, setActiveSessionId] = useState<string>('');
  
  // 流式连接状态
  const [isSSEConnected, setIsSSEConnected] = useState(false);
  const [isWSConnected, setIsWSConnected] = useState(false);
  const [streamingEvents, setStreamingEvents] = useState<string[]>([]);
  const [wsMessages, setWsMessages] = useState<string[]>([]);
  
  // EventSource 和 WebSocket 引用
  const sseRef = useRef<EventSource | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  
  // 会话创建表单
  const [sessionForm, setSessionForm] = useState({
    agent_id: 'streaming_agent',
    message: 'Hello, streaming world!',
    buffer_size: 1024
  });

  // 获取系统指标
  const fetchSystemMetrics = async () => {
    try {
      const response = await streamingService.getSystemMetrics();
      setSystemMetrics(response.system_metrics);
    } catch (err) {
      logger.error('获取系统指标失败:', err);
    }
  };

  // 获取背压状态
  const fetchBackpressureStatus = async () => {
    try {
      const response = await streamingService.getBackpressureStatus();
      setBackpressureStatus(response.backpressure_status || null);
    } catch (err) {
      logger.error('获取背压状态失败:', err);
    }
  };

  // 获取队列状态
  const fetchQueueStatus = async () => {
    try {
      const response = await streamingService.getQueueStatus();
      setQueueStatus(response);
    } catch (err) {
      logger.error('获取队列状态失败:', err);
    }
  };

  // 获取会话列表
  const fetchSessions = async () => {
    try {
      const response = await streamingService.getSessions();
      setSessions(Object.values(response.sessions || {}));
    } catch (err) {
      logger.error('获取会话列表失败:', err);
    }
  };

  // 获取健康状态
  const fetchHealthStatus = async () => {
    try {
      const response = await streamingService.getHealthStatus();
      setHealthStatus(response);
    } catch (err) {
      logger.error('获取健康状态失败:', err);
    }
  };

  // 创建会话
  const handleCreateSession = async () => {
    try {
      setLoading(true);
      const response = await streamingService.createSession(sessionForm);
      message.success(`会话创建成功: ${response.session_id}`);
      setActiveSessionId(response.session_id);
      await fetchSessions();
    } catch (err) {
      message.error('会话创建失败: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  // 停止会话
  const handleStopSession = async (sessionId: string) => {
    try {
      await streamingService.stopSession(sessionId);
      message.success('会话已停止');
      await fetchSessions();
    } catch (err) {
      message.error('停止会话失败: ' + (err as Error).message);
    }
  };

  // 启动SSE连接
  const startSSEConnection = () => {
    if (sseRef.current) {
      sseRef.current.close();
    }
    const sessionId = activeSessionId || sessions[0]?.session_id;
    if (!sessionId) {
      message.error('请先创建会话');
      return;
    }
    const eventSource = streamingService.createSSEConnection(sessionId, sessionForm.message);
    
    eventSource.onopen = () => {
      setIsSSEConnected(true);
      message.success('SSE连接已建立');
    };
    
    eventSource.onmessage = (event) => {
      let parsed: any = null;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        parsed = null;
      }
      if (parsed?.type === 'error') {
        eventSource.close();
        sseRef.current = null;
        setIsSSEConnected(false);
        const errorMsg = typeof parsed.data === 'string' ? parsed.data : parsed.data?.message;
        message.error(errorMsg || 'SSE连接错误');
        return;
      }
      setStreamingEvents(prev => [...prev.slice(-19), `[${new Date().toLocaleTimeString()}] ${event.data}`]);
    };
    
    eventSource.onerror = () => {
      eventSource.close();
      sseRef.current = null;
      setIsSSEConnected(false);
      message.error('SSE连接错误');
    };
    
    sseRef.current = eventSource;
  };

  // 停止SSE连接
  const stopSSEConnection = () => {
    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
      setIsSSEConnected(false);
      message.info('SSE连接已关闭');
    }
  };

  // 启动WebSocket连接
  const startWSConnection = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    const sessionId = activeSessionId || sessions[0]?.session_id;
    if (!sessionId) {
      message.error('请先创建会话');
      return;
    }
    const websocket = streamingService.createWebSocketConnection(sessionId);
    
    websocket.onopen = () => {
      setIsWSConnected(true);
      message.success('WebSocket连接已建立');
      websocket.send(JSON.stringify({
        type: 'chat',
        data: { agent_id: sessionForm.agent_id, message: sessionForm.message }
      }));
    };
    
    websocket.onmessage = (event) => {
      let parsed: any = null;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        parsed = null;
      }
      if (parsed?.type === 'error') {
        websocket.close();
        const errorMsg = typeof parsed.data === 'string' ? parsed.data : parsed.data?.message;
        message.error(errorMsg || 'WebSocket连接错误');
        return;
      }
      setWsMessages(prev => [...prev.slice(-19), `[${new Date().toLocaleTimeString()}] ${event.data}`]);
    };
    
    websocket.onerror = () => {
      setIsWSConnected(false);
      websocket.close();
      message.error('WebSocket连接错误');
    };
    
    websocket.onclose = () => {
      setIsWSConnected(false);
      wsRef.current = null;
      message.info('WebSocket连接已关闭');
    };
    
    wsRef.current = websocket;
  };

  // 停止WebSocket连接
  const stopWSConnection = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setIsWSConnected(false);
      message.info('WebSocket连接已关闭');
    }
  };

  // 定时刷新数据
  useEffect(() => {
    const fetchData = async () => {
      await Promise.all([
        fetchSystemMetrics(),
        fetchBackpressureStatus(),
        fetchQueueStatus(),
        fetchSessions(),
        fetchHealthStatus()
      ]);
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // 每5秒刷新一次

    return () => {
      clearInterval(interval);
      if (sseRef.current) {
        sseRef.current.close();
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // 会话表格列定义
  const sessionColumns = [
    {
      title: '会话ID',
      dataIndex: 'session_id',
      key: 'session_id',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'active' ? 'green' : 'orange'}>{status}</Tag>
      ),
    },
    {
      title: '持续时间',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration: number) => duration ? `${duration}s` : 'N/A',
    },
    {
      title: '令牌数',
      dataIndex: 'token_count',
      key: 'token_count',
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: any) => (
        <Space>
          <Button size="small" danger onClick={() => handleStopSession(record.session_id)}>
            停止
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>流式处理监控</Title>

      {error && (
        <Alert
          message="错误"
          description={error}
          type="error"
          closable
          style={{ marginBottom: 16 }}
          onClose={() => setError(null)}
        />
      )}

      {success && (
        <Alert
          message="成功"
          description={success}
          type="success"
          closable
          style={{ marginBottom: 16 }}
          onClose={() => setSuccess(null)}
        />
      )}

      <Tabs defaultActiveKey="metrics" type="card" items={[
        {
          key: 'metrics',
          label: '系统指标',
          children: (
            <Row gutter={[16, 16]}>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="活跃会话"
                    value={systemMetrics?.active_sessions || 0}
                    prefix={<ThunderboltOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="总会话数"
                    value={systemMetrics?.total_sessions_created || 0}
                    prefix={<MonitorOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="处理的令牌数"
                    value={systemMetrics?.total_tokens_processed || 0}
                    prefix={<ApiOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="缓冲区使用量"
                    value={systemMetrics?.total_buffer_usage || 0}
                    suffix="KB"
                  />
                </Card>
              </Col>
              
              {healthStatus && (
                <Col span={24}>
                  <Card title="健康状态">
                    <Row gutter={[16, 16]}>
                      <Col span={6}>
                        <Statistic
                          title="服务状态"
                          value={healthStatus.status}
                          valueStyle={{ color: healthStatus.status === 'healthy' ? '#3f8600' : '#cf1322' }}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="运行时间"
                          value={Math.floor(healthStatus.uptime / 60)}
                          suffix="分钟"
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="活跃会话"
                          value={healthStatus.active_sessions}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="总会话数"
                          value={healthStatus.total_sessions}
                        />
                      </Col>
                    </Row>
                  </Card>
                </Col>
              )}

              <Col span={24}>
                <Card title="流控状态">
                  <Space wrap>
                    <Tag color={backpressureStatus ? 'green' : 'default'}>
                      {backpressureStatus ? `背压: ${backpressureStatus.throttle_level}` : '背压: 未启用'}
                    </Tag>
                    <Tag>
                      队列利用率: {queueStatus ? (queueStatus.system_summary.system_utilization * 100).toFixed(1) : '0.0'}%
                    </Tag>
                    <Tag color={queueStatus?.overloaded_queues.length ? 'red' : 'green'}>
                      过载队列: {queueStatus?.overloaded_queues.length || 0}
                    </Tag>
                  </Space>
                </Card>
              </Col>

              <Col span={24}>
                <Card
                  title="会话管理"
                  extra={
                    <Button type="primary" onClick={handleCreateSession} loading={loading}>
                      创建会话
                    </Button>
                  }
                >
                  <Tag color={activeSessionId ? 'green' : 'red'}>
                    {activeSessionId ? `当前会话: ${activeSessionId}` : '未创建会话'}
                  </Tag>
                </Card>
              </Col>

              <Col span={24}>
                <Card title="会话列表">
                  <Table dataSource={sessions} columns={sessionColumns} rowKey="session_id" pagination={false} />
                </Card>
              </Col>
            </Row>
          )
        },
        {
          key: 'streaming',
          label: '流式连接',
          children: (
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card 
                  title="Server-Sent Events (SSE)" 
                  extra={
                    <Space>
                      <Button
                        type={isSSEConnected ? "default" : "primary"}
                        icon={<PlayCircleOutlined />}
                        onClick={startSSEConnection}
                        disabled={isSSEConnected}
                      >
                        连接
                      </Button>
                      <Button
                        danger
                        icon={<StopOutlined />}
                        onClick={stopSSEConnection}
                        disabled={!isSSEConnected}
                      >
                        断开
                      </Button>
                    </Space>
                  }
                >
                  <div style={{ marginBottom: 16 }}>
                    <Tag color={isSSEConnected ? 'green' : 'red'}>
                      {isSSEConnected ? '已连接' : '未连接'}
                    </Tag>
                  </div>
                  <div style={{ height: 300, overflowY: 'auto', backgroundColor: '#f5f5f5', padding: 8, borderRadius: 4 }}>
                    {streamingEvents.map((event, index) => (
                      <div key={index} style={{ fontSize: '12px', marginBottom: 4 }}>
                        {event}
                      </div>
                    ))}
                    {streamingEvents.length === 0 && (
                      <div style={{ color: '#999', textAlign: 'center', paddingTop: 100 }}>
                        等待事件数据...
                      </div>
                    )}
                  </div>
                </Card>
              </Col>
              
              <Col span={12}>
                <Card 
                  title="WebSocket连接" 
                  extra={
                    <Space>
                      <Button
                        type={isWSConnected ? "default" : "primary"}
                        icon={<PlayCircleOutlined />}
                        onClick={startWSConnection}
                        disabled={isWSConnected}
                      >
                        连接
                      </Button>
                      <Button
                        danger
                        icon={<StopOutlined />}
                        onClick={stopWSConnection}
                        disabled={!isWSConnected}
                      >
                        断开
                      </Button>
                    </Space>
                  }
                >
                  <div style={{ marginBottom: 16 }}>
                    <Tag color={isWSConnected ? 'green' : 'red'}>
                      {isWSConnected ? '已连接' : '未连接'}
                    </Tag>
                  </div>
                  <div style={{ height: 300, overflowY: 'auto', backgroundColor: '#f5f5f5', padding: 8, borderRadius: 4 }}>
                    {wsMessages.map((message, index) => (
                      <div key={index} style={{ fontSize: '12px', marginBottom: 4 }}>
                        {message}
                      </div>
                    ))}
                    {wsMessages.length === 0 && (
                      <div style={{ color: '#999', textAlign: 'center', paddingTop: 100 }}>
                        等待WebSocket消息...
                      </div>
                    )}
                  </div>
                </Card>
              </Col>
            </Row>
          )
        }
      ]} />
    </div>
  )
};

export default StreamingMonitorPage;
