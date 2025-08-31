import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/Tabs';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription } from '../components/ui/alert';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { 
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

interface SecurityMetrics {
  authentication: {
    total_attempts_24h: number;
    successful_attempts_24h: number;
    failed_attempts_24h: number;
    success_rate: number;
    average_response_time_ms: number;
  };
  access_control: {
    total_requests_24h: number;
    granted_requests_24h: number;
    denied_requests_24h: number;
    approval_rate: number;
    policy_evaluation_time_ms: number;
  };
  communication: {
    active_sessions: number;
    total_messages_24h: number;
    encryption_overhead_ms: number;
    integrity_violations: number;
  };
  threat_detection: {
    events_processed_24h: number;
    threats_detected_24h: number;
    false_positives_24h: number;
    alert_response_time_minutes: number;
  };
}

interface SecurityAlert {
  id: string;
  alert_type: string;
  title: string;
  threat_level: 'low' | 'medium' | 'high' | 'critical';
  status: 'active' | 'acknowledged' | 'resolved';
  agent_id?: string;
  created_at: string;
  indicators: Record<string, any>;
}

interface AgentIdentity {
  id: string;
  agent_id: string;
  display_name: string;
  trust_score: number;
  last_authentication: string;
  failed_attempts: number;
  is_locked: boolean;
  authentication_methods: string[];
}

const DistributedSecurityMonitorPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [metrics, setMetrics] = useState<SecurityMetrics | null>(null);
  const [alerts, setAlerts] = useState<SecurityAlert[]>([]);
  const [agents, setAgents] = useState<AgentIdentity[]>([]);
  const [loading, setLoading] = useState(true);

  // 模拟数据加载
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // 模拟安全指标数据
      const mockMetrics: SecurityMetrics = {
        authentication: {
          total_attempts_24h: 2450,
          successful_attempts_24h: 2380,
          failed_attempts_24h: 70,
          success_rate: 0.971,
          average_response_time_ms: 85.4
        },
        access_control: {
          total_requests_24h: 15200,
          granted_requests_24h: 14650,
          denied_requests_24h: 550,
          approval_rate: 0.964,
          policy_evaluation_time_ms: 12.3
        },
        communication: {
          active_sessions: 156,
          total_messages_24h: 45000,
          encryption_overhead_ms: 3.2,
          integrity_violations: 0
        },
        threat_detection: {
          events_processed_24h: 28500,
          threats_detected_24h: 12,
          false_positives_24h: 4,
          alert_response_time_minutes: 8.5
        }
      };

      // 模拟安全告警数据
      const mockAlerts: SecurityAlert[] = [
        {
          id: 'alert_001',
          alert_type: 'brute_force_attack',
          title: '检测到暴力破解攻击',
          threat_level: 'high',
          status: 'active',
          agent_id: 'agent_007',
          created_at: '2024-01-20T10:30:00Z',
          indicators: { failed_attempts: 8, source_ip: '192.168.1.100' }
        },
        {
          id: 'alert_002',
          alert_type: 'privilege_escalation',
          title: '可疑权限提升行为',
          threat_level: 'medium',
          status: 'acknowledged',
          agent_id: 'agent_012',
          created_at: '2024-01-20T09:15:00Z',
          indicators: { escalation_attempts: 3, target_resources: ['admin_api'] }
        },
        {
          id: 'alert_003',
          alert_type: 'anomaly_detection',
          title: '异常访问模式',
          threat_level: 'low',
          status: 'resolved',
          agent_id: 'agent_023',
          created_at: '2024-01-20T08:45:00Z',
          indicators: { unusual_requests: 45, time_pattern: 'after_hours' }
        }
      ];

      // 模拟智能体身份数据
      const mockAgents: AgentIdentity[] = [
        {
          id: 'id_001',
          agent_id: 'agent_001',
          display_name: '数据分析智能体',
          trust_score: 98.5,
          last_authentication: '2024-01-20T10:45:00Z',
          failed_attempts: 0,
          is_locked: false,
          authentication_methods: ['pki_certificate', 'mfa']
        },
        {
          id: 'id_002',
          agent_id: 'agent_007',
          display_name: '网络监控智能体',
          trust_score: 76.2,
          last_authentication: '2024-01-20T10:20:00Z',
          failed_attempts: 8,
          is_locked: true,
          authentication_methods: ['pki_certificate']
        },
        {
          id: 'id_003',
          agent_id: 'agent_012',
          display_name: '任务调度智能体',
          trust_score: 89.7,
          last_authentication: '2024-01-20T10:40:00Z',
          failed_attempts: 2,
          is_locked: false,
          authentication_methods: ['pki_certificate', 'mfa', 'biometric']
        }
      ];

      setMetrics(mockMetrics);
      setAlerts(mockAlerts);
      setAgents(mockAgents);
      setLoading(false);
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

  // 模拟24小时趋势数据
  const generateTrendData = () => {
    return Array.from({ length: 24 }, (_, i) => ({
      time: `${i}:00`,
      auth_success: Math.floor(Math.random() * 100) + 80,
      auth_failed: Math.floor(Math.random() * 10) + 2,
      access_granted: Math.floor(Math.random() * 500) + 400,
      access_denied: Math.floor(Math.random() * 50) + 10,
      threats: Math.floor(Math.random() * 3),
      sessions: Math.floor(Math.random() * 20) + 100
    }));
  };

  const trendData = generateTrendData();

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