import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react';
import {
import { logger } from '../utils/logger'
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Tag,
  List,
  Avatar,
  Divider,
  message,
  Spin,
  Alert,
  Badge
} from 'antd';
import {
  RobotOutlined,
  TeamOutlined,
  PlayCircleOutlined,
  SyncOutlined,
  CheckCircleOutlined,
  ApiOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;

interface Agent {
  id: string;
  name: string;
  role: string;
  status: string;
  capabilities: string[];
  last_active: string;
}

interface AgentSystemStats {
  total_agents: number;
  active_agents: number;
  total_conversations: number;
  system_status: string;
}

const SimpleMultiAgentPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [stats, setStats] = useState<AgentSystemStats>({
    total_agents: 0,
    active_agents: 0,
    total_conversations: 0,
    system_status: 'unknown'
  });

  useEffect(() => {
    loadAgents();
  }, []);

  const loadAgents = async () => {
    setLoading(true);
    try {
      const response = await apiFetch(buildApiUrl('/api/v1/multi-agent/agents'));
      const data = await response.json();
      
      if (data.success && data.data && data.data.agents) {
        setAgents(data.data.agents);
        setStats({
          total_agents: data.data.agents.length,
          active_agents: data.data.agents.filter((a: Agent) => a.status === 'active').length,
          total_conversations: data.data.total || 0,
          system_status: 'healthy'
        });
        message.success(`成功加载 ${data.data.agents.length} 个智能代理`);
      } else {
        message.error('加载多代理系统失败');
      }
    } catch (error) {
      logger.error('API调用失败:', error);
      message.error('连接服务器失败');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    const colors: { [key: string]: string } = {
      'active': 'green',
      'inactive': 'default',
      'busy': 'orange',
      'error': 'red'
    };
    return colors[status] || 'default';
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircleOutlined />;
      case 'busy': return <SyncOutlined spin />;
      default: return <PlayCircleOutlined />;
    }
  };

  const getRoleColor = (role: string) => {
    const colors: { [key: string]: string } = {
      'researcher': 'blue',
      'coder': 'green',
      'manager': 'orange',
      'analyst': 'purple'
    };
    return colors[role] || 'default';
  };

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <TeamOutlined style={{ color: '#1890ff' }} /> 多智能体协作系统
      </Title>
      <Text type="secondary">AutoGen v0.4 多代理协作平台</Text>

      <Divider />

      {/* 系统统计 */}
      <Card style={{ marginBottom: '24px' }}>
        <Row gutter={16}>
          <Col span={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Badge status="processing" />
              <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
                {stats.total_agents}
              </Title>
              <Text type="secondary">总代理数</Text>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Badge status="success" />
              <Title level={3} style={{ margin: 0, color: '#52c41a' }}>
                {stats.active_agents}
              </Title>
              <Text type="secondary">活跃代理</Text>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Badge status="default" />
              <Title level={3} style={{ margin: 0, color: '#722ed1' }}>
                {stats.total_conversations}
              </Title>
              <Text type="secondary">总对话数</Text>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Badge status={stats.system_status === 'healthy' ? 'success' : 'error'} />
              <Title level={4} style={{ margin: 0, color: '#f5222d' }}>
                {stats.system_status}
              </Title>
              <Text type="secondary">系统状态</Text>
            </Card>
          </Col>
        </Row>
      </Card>

      {/* 操作按钮 */}
      <Card style={{ marginBottom: '24px' }}>
        <Space>
          <Button type="primary" icon={<SyncOutlined />} onClick={loadAgents} loading={loading}>
            刷新代理状态
          </Button>
          <Button icon={<ApiOutlined />}>
            API健康检查
          </Button>
          <Button icon={<PlayCircleOutlined />}>
            启动对话会话
          </Button>
        </Space>
      </Card>

      {/* 代理列表 */}
      <Card title="智能代理列表" loading={loading}>
        {agents.length === 0 && !loading ? (
          <Alert
            message="暂无代理数据"
            description="尝试刷新页面或检查API连接状态"
            type="info"
            showIcon
          />
        ) : (
          <List
            itemLayout="vertical"
            dataSource={agents}
            renderItem={(agent) => (
              <List.Item
                key={agent.id}
                extra={
                  <Space direction="vertical" align="end">
                    <Tag color={getStatusColor(agent.status)} icon={getStatusIcon(agent.status)}>
                      {agent.status}
                    </Tag>
                    <Tag color={getRoleColor(agent.role)}>
                      {agent.role}
                    </Tag>
                  </Space>
                }
              >
                <List.Item.Meta
                  avatar={<Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#1890ff' }} />}
                  title={
                    <Space>
                      <Text strong>{agent.name}</Text>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        #{agent.id}
                      </Text>
                    </Space>
                  }
                  description={
                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                      <div>
                        <Text strong>能力: </Text>
                        {agent.capabilities.map((capability, index) => (
                          <Tag key={index} size="small" color="blue">
                            {capability}
                          </Tag>
                        ))}
                      </div>
                      <div>
                        <Text strong>最后活跃: </Text>
                        <Text type="secondary">
                          {new Date(agent.last_active).toLocaleString('zh-CN')}
                        </Text>
                      </div>
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        )}
      </Card>

      {/* 系统能力概览 */}
      {agents.length > 0 && (
        <Card title="系统能力概览" style={{ marginTop: '24px' }}>
          <Row gutter={[16, 16]}>
            {Array.from(new Set(agents.flatMap(a => a.capabilities))).map((capability) => (
              <Col key={capability} span={8}>
                <Card size="small" style={{ textAlign: 'center' }}>
                  <Tag color="blue" style={{ marginBottom: '8px' }}>
                    {capability}
                  </Tag>
                  <div>
                    <Text strong>
                      {agents.filter(a => a.capabilities.includes(capability)).length}
                    </Text>
                    <Text type="secondary"> 个代理支持</Text>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </Card>
      )}
    </div>
  );
};

export default SimpleMultiAgentPage;
