import React from 'react'
import {
  Card,
  Typography,
  Space,
  Button,
  Row,
  Col,
  Statistic,
  Progress,
  Tag,
} from 'antd'
import {
  CloudServerOutlined,
  SettingOutlined,
  BugOutlined,
  MonitorOutlined,
  SecurityScanOutlined,
  DatabaseOutlined,
  RocketOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'

const { Title, Text } = Typography

const EnterpriseArchitecturePageSimple: React.FC = () => {
  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title
          level={2}
          style={{ margin: 0, display: 'flex', alignItems: 'center' }}
        >
          <CloudServerOutlined
            style={{ marginRight: '12px', color: '#1890ff' }}
          />
          企业架构管理总览
        </Title>
        <Text type="secondary">统一管理AI智能体系统的企业级架构组件</Text>
      </div>

      {/* 系统状态统计 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="服务总数"
              value={28}
              prefix={<CloudServerOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
            <div style={{ marginTop: '8px' }}>
              <Tag color="green">运行中: 26</Tag>
              <Tag color="orange">维护中: 2</Tag>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="配置项目"
              value={156}
              prefix={<SettingOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
            <div style={{ marginTop: '8px' }}>
              <Progress percent={92} size="small" showInfo={false} />
              <Text type="secondary">配置完整度 92%</Text>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="系统健康度"
              value={98.5}
              precision={1}
              suffix="%"
              prefix={<MonitorOutlined />}
              valueStyle={{ color: '#13c2c2' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="安全评分"
              value={95}
              suffix="/100"
              prefix={<SecurityScanOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 架构组件模块 */}
      <Row gutter={[16, 16]}>
        <Col span={8}>
          <Card
            title="核心架构组件"
            extra={
              <Button type="primary" size="small">
                管理
              </Button>
            }
            style={{ height: '280px' }}
          >
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <div>
                <Text strong>🏗️ 微服务架构</Text>
                <br />
                <Text type="secondary">28个微服务，容器化部署</Text>
                <Progress percent={95} size="small" />
              </div>

              <div>
                <Text strong>🔗 API网关</Text>
                <br />
                <Text type="secondary">统一入口，负载均衡</Text>
                <Progress percent={98} size="small" />
              </div>

              <div>
                <Text strong>📊 监控中心</Text>
                <br />
                <Text type="secondary">实时监控，告警管理</Text>
                <Progress percent={89} size="small" />
              </div>
            </Space>
          </Card>
        </Col>

        <Col span={8}>
          <Card
            title="配置管理中心"
            extra={
              <Button type="primary" size="small">
                配置
              </Button>
            }
            style={{ height: '280px' }}
          >
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <div>
                <Text strong>⚙️ 环境配置</Text>
                <br />
                <Text type="secondary">开发/测试/生产环境</Text>
                <div style={{ marginTop: '4px' }}>
                  <Tag color="blue">Dev</Tag>
                  <Tag color="orange">Test</Tag>
                  <Tag color="green">Prod</Tag>
                </div>
              </div>

              <div>
                <Text strong>🔐 安全配置</Text>
                <br />
                <Text type="secondary">访问控制，权限管理</Text>
                <Progress percent={96} size="small" />
              </div>

              <div>
                <Text strong>📋 应用配置</Text>
                <br />
                <Text type="secondary">动态配置，热更新</Text>
                <Progress percent={88} size="small" />
              </div>
            </Space>
          </Card>
        </Col>

        <Col span={8}>
          <Card
            title="调试工具集"
            extra={
              <Button type="primary" size="small">
                调试
              </Button>
            }
            style={{ height: '280px' }}
          >
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <div>
                <Text strong>🐛 错误追踪</Text>
                <br />
                <Text type="secondary">分布式链路追踪</Text>
                <Tag color="green">正常</Tag>
              </div>

              <div>
                <Text strong>📈 性能分析</Text>
                <br />
                <Text type="secondary">响应时间，吞吐量分析</Text>
                <Tag color="blue">运行中</Tag>
              </div>

              <div>
                <Text strong>💾 日志管理</Text>
                <br />
                <Text type="secondary">集中化日志收集分析</Text>
                <Tag color="orange">24.5GB</Tag>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 基础设施状态 */}
      <Card title="基础设施状态" style={{ marginTop: '24px' }}>
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <div style={{ textAlign: 'center' }}>
              <DatabaseOutlined
                style={{
                  fontSize: '32px',
                  color: '#1890ff',
                  marginBottom: '8px',
                }}
              />
              <div>
                <Text strong>数据存储</Text>
              </div>
              <div>
                <Text type="secondary">PostgreSQL + Redis</Text>
              </div>
              <Tag color="green">运行正常</Tag>
            </div>
          </Col>

          <Col span={6}>
            <div style={{ textAlign: 'center' }}>
              <RocketOutlined
                style={{
                  fontSize: '32px',
                  color: '#52c41a',
                  marginBottom: '8px',
                }}
              />
              <div>
                <Text strong>容器编排</Text>
              </div>
              <div>
                <Text type="secondary">Kubernetes</Text>
              </div>
              <Tag color="blue">28 Pods</Tag>
            </div>
          </Col>

          <Col span={6}>
            <div style={{ textAlign: 'center' }}>
              <MonitorOutlined
                style={{
                  fontSize: '32px',
                  color: '#faad14',
                  marginBottom: '8px',
                }}
              />
              <div>
                <Text strong>监控告警</Text>
              </div>
              <div>
                <Text type="secondary">Prometheus + Grafana</Text>
              </div>
              <Tag color="green">正常监控</Tag>
            </div>
          </Col>

          <Col span={6}>
            <div style={{ textAlign: 'center' }}>
              <ThunderboltOutlined
                style={{
                  fontSize: '32px',
                  color: '#722ed1',
                  marginBottom: '8px',
                }}
              />
              <div>
                <Text strong>消息队列</Text>
              </div>
              <div>
                <Text type="secondary">NATS Streaming</Text>
              </div>
              <Tag color="processing">处理中</Tag>
            </div>
          </Col>
        </Row>
      </Card>

      {/* 快速操作 */}
      <Card title="快速操作" style={{ marginTop: '24px' }}>
        <Space wrap>
          <Button type="primary" icon={<SettingOutlined />}>
            系统配置
          </Button>
          <Button icon={<MonitorOutlined />}>监控面板</Button>
          <Button icon={<BugOutlined />}>问题诊断</Button>
          <Button icon={<SecurityScanOutlined />}>安全检查</Button>
          <Button icon={<DatabaseOutlined />}>数据库管理</Button>
        </Space>
      </Card>
    </div>
  )
}

export default EnterpriseArchitecturePageSimple
