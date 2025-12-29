import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription } from '../components/ui/alert';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { 
import { logger } from '../utils/logger'
  Shield, 
  AlertTriangle, 
  Users, 
  Lock, 
  Activity, 
  TrendingUp,
  Eye,
  Key,
  Ban,
  CheckCircle,
  XCircle,
  Clock,
  Globe,
  Server,
  Database
} from 'lucide-react';
import { distributedSecurityService, type SecurityMetrics, type SecurityAlert, type AgentIdentity } from '../services/distributedSecurityService';
import { message } from 'antd';

const DistributedSecurityMonitorPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [metrics, setMetrics] = useState<SecurityMetrics | null>(null);
  const [alerts, setAlerts] = useState<SecurityAlert[]>([]);
  const [agents, setAgents] = useState<AgentIdentity[]>([]);
  const [loading, setLoading] = useState(true);
  const [metricsHistory, setMetricsHistory] = useState<SecurityMetrics[]>([]);

  // 加载数据
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      
      try {
        const now = new Date();
        const start = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        const [metricsData, alertsData, agentsData, historyData] = await Promise.all([
          distributedSecurityService.getMetrics(),
          distributedSecurityService.getAlerts({ limit: 10 }),
          distributedSecurityService.getAgents(),
          distributedSecurityService.getMetricsHistory({
            start: start.toISOString(),
            end: now.toISOString()
          })
        ]);
        
        setMetrics(metricsData);
        setAlerts(alertsData);
        setAgents(agentsData);
        setMetricsHistory(historyData || []);
      } catch (error) {
        logger.error('加载安全数据失败:', error);
        message.error('加载安全监控数据失败');
        setMetrics(null);
        setAlerts([]);
        setAgents([]);
        setMetricsHistory([]);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // 安全等级颜色映射
  const getThreatLevelColor = (level: string) => {
    switch (level) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-200';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-blue-100 text-blue-800 border-blue-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  // 状态颜色映射
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-red-100 text-red-800';
      case 'acknowledged': return 'bg-yellow-100 text-yellow-800';
      case 'resolved': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const trendData = (metricsHistory && metricsHistory.length > 0
    ? metricsHistory
    : metrics
      ? [metrics]
      : []
  ).map((item, index) => ({
    time: `${index}`,
    auth_success: item.authentication.successful_attempts_24h,
    auth_failed: item.authentication.failed_attempts_24h,
    access_granted: item.access_control.granted_requests_24h,
    access_denied: item.access_control.denied_requests_24h,
    threats: item.threat_detection.threats_detected_24h,
    sessions: item.communication.active_sessions
  }));

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg">加载安全监控数据中...</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* 页面头部 */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Shield className="h-8 w-8 text-blue-600" />
          分布式安全框架监控
        </h1>
        <p className="text-gray-600 mt-2">
          实时监控智能体身份认证、访问控制、加密通信和威胁检测状态
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">安全概览</TabsTrigger>
          <TabsTrigger value="authentication">身份认证</TabsTrigger>
          <TabsTrigger value="access-control">访问控制</TabsTrigger>
          <TabsTrigger value="communication">加密通信</TabsTrigger>
          <TabsTrigger value="threats">威胁监控</TabsTrigger>
        </TabsList>

        {/* 安全概览 */}
        <TabsContent value="overview" className="space-y-6">
          {/* 核心指标卡片 */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">活跃智能体</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {agents.filter(a => !a.is_locked).length}
                </div>
                <p className="text-xs text-muted-foreground">
                  总计 {agents.length} 个智能体
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">认证成功率</CardTitle>
                <CheckCircle className="h-4 w-4 text-green-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {((metrics?.authentication.success_rate || 0) * 100).toFixed(1)}%
                </div>
                <p className="text-xs text-muted-foreground">
                  24小时内 {metrics?.authentication.total_attempts_24h} 次尝试
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">活跃会话</CardTitle>
                <Globe className="h-4 w-4 text-blue-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {metrics?.communication.active_sessions}
                </div>
                <p className="text-xs text-muted-foreground">
                  加密通信会话
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">安全威胁</CardTitle>
                <AlertTriangle className="h-4 w-4 text-red-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {alerts.filter(a => a.status === 'active').length}
                </div>
                <p className="text-xs text-muted-foreground">
                  活跃告警
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 24小时趋势图表 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>认证趋势 (24小时)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Area
                      type="monotone"
                      dataKey="auth_success"
                      stackId="1"
                      stroke="#10b981"
                      fill="#10b981"
                      fillOpacity={0.6}
                      name="成功认证"
                    />
                    <Area
                      type="monotone"
                      dataKey="auth_failed"
                      stackId="1"
                      stroke="#ef4444"
                      fill="#ef4444"
                      fillOpacity={0.6}
                      name="失败认证"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>访问控制趋势 (24小时)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="access_granted"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      name="允许访问"
                    />
                    <Line
                      type="monotone"
                      dataKey="access_denied"
                      stroke="#ef4444"
                      strokeWidth={2}
                      name="拒绝访问"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* 安全告警列表 */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5" />
                最新安全告警
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {alerts.slice(0, 5).map((alert) => (
                  <div key={alert.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        <Badge className={getThreatLevelColor(alert.threat_level)}>
                          {alert.threat_level.toUpperCase()}
                        </Badge>
                        <Badge className={getStatusColor(alert.status)}>
                          {alert.status}
                        </Badge>
                      </div>
                      <div>
                        <h4 className="font-semibold">{alert.title}</h4>
                        <p className="text-sm text-gray-600">
                          智能体: {alert.agent_id} | 
                          时间: {new Date(alert.created_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {alert.status === 'active' && (
                        <Button variant="outline" size="sm">
                          处理
                        </Button>
                      )}
                      <Button variant="ghost" size="sm">
                        <Eye className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* 身份认证监控 */}
        <TabsContent value="authentication" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>智能体身份状态</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {agents.map((agent) => (
                    <div key={agent.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center gap-4">
                        <div className={`w-3 h-3 rounded-full ${agent.is_locked ? 'bg-red-500' : 'bg-green-500'}`} />
                        <div>
                          <h4 className="font-semibold">{agent.display_name}</h4>
                          <p className="text-sm text-gray-600">ID: {agent.agent_id}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-sm">信任分数:</span>
                          <Badge variant={agent.trust_score >= 90 ? "default" : agent.trust_score >= 80 ? "secondary" : "destructive"}>
                            {agent.trust_score.toFixed(1)}
                          </Badge>
                        </div>
                        <div className="text-xs text-gray-500">
                          失败次数: {agent.failed_attempts}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>认证方法分布</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'PKI证书', value: 65, fill: '#3b82f6' },
                        { name: 'MFA', value: 25, fill: '#10b981' },
                        { name: '生物识别', value: 10, fill: '#f59e0b' }
                      ]}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({name, value}) => `${name}: ${value}%`}
                    />
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* 访问控制监控 */}
        <TabsContent value="access-control" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>访问请求分析</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={trendData.slice(12, 24)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="access_granted" fill="#10b981" name="允许" />
                    <Bar dataKey="access_denied" fill="#ef4444" name="拒绝" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>策略执行统计</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                    <span className="font-medium">RBAC策略</span>
                    <Badge className="bg-green-100 text-green-800">75% 匹配</Badge>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                    <span className="font-medium">ABAC策略</span>
                    <Badge className="bg-blue-100 text-blue-800">20% 匹配</Badge>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span className="font-medium">默认拒绝</span>
                    <Badge className="bg-gray-100 text-gray-800">5% 匹配</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* 加密通信监控 */}
        <TabsContent value="communication" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>通信会话统计</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span>活跃会话</span>
                    <Badge className="bg-green-100 text-green-800">
                      {metrics?.communication.active_sessions}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>24小时消息总数</span>
                    <Badge className="bg-blue-100 text-blue-800">
                      {metrics?.communication.total_messages_24h.toLocaleString()}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>加密开销</span>
                    <Badge className="bg-yellow-100 text-yellow-800">
                      {metrics?.communication.encryption_overhead_ms}ms
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>完整性违规</span>
                    <Badge className="bg-red-100 text-red-800">
                      {metrics?.communication.integrity_violations}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>加密算法分布</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'AES-256-GCM', value: 85, fill: '#3b82f6' },
                        { name: 'ChaCha20-Poly1305', value: 12, fill: '#10b981' },
                        { name: '其他', value: 3, fill: '#f59e0b' }
                      ]}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({name, value}) => `${name}: ${value}%`}
                    />
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* 威胁监控 */}
        <TabsContent value="threats" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>威胁检测时间线</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="threats"
                      stroke="#ef4444"
                      strokeWidth={2}
                      name="威胁检测"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>威胁类型分布</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">暴力破解</span>
                    <Badge className="bg-red-100 text-red-800">45%</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">权限提升</span>
                    <Badge className="bg-orange-100 text-orange-800">25%</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">异常行为</span>
                    <Badge className="bg-yellow-100 text-yellow-800">20%</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">其他</span>
                    <Badge className="bg-gray-100 text-gray-800">10%</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 威胁告警详情 */}
          <Card>
            <CardHeader>
              <CardTitle>威胁告警详情</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {alerts.map((alert) => (
                  <div key={alert.id} className="p-4 border rounded-lg">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-semibold">{alert.title}</h4>
                        <p className="text-sm text-gray-600">类型: {alert.alert_type}</p>
                      </div>
                      <div className="flex gap-2">
                        <Badge className={getThreatLevelColor(alert.threat_level)}>
                          {alert.threat_level}
                        </Badge>
                        <Badge className={getStatusColor(alert.status)}>
                          {alert.status}
                        </Badge>
                      </div>
                    </div>
                    <div className="text-sm text-gray-600 mb-2">
                      <p>智能体: {alert.agent_id}</p>
                      <p>时间: {new Date(alert.created_at).toLocaleString()}</p>
                    </div>
                    <div className="bg-gray-50 p-2 rounded text-xs">
                      <strong>指标:</strong> {JSON.stringify(alert.indicators)}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default DistributedSecurityMonitorPage;
