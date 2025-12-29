import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '../components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Input } from '../components/ui/input';
import { Textarea } from '../components/ui/textarea';
import { Alert, AlertDescription } from '../components/ui/alert';
import { Progress } from '../components/ui/progress';
import { Separator } from '../components/ui/separator';
import { 
  Shield, Settings, FileCheck, Users, Lock, 
  AlertTriangle, Eye, Key, Network, Download,
  CheckCircle, XCircle, Clock, Zap 
} from 'lucide-react';
import { securityEnhancedService } from '../services/securityEnhancedService';

const SecurityManagementEnhancedPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('config');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // 状态管理
  const [configData, setConfigData] = useState<any>(null);
  const [mcpTools, setMcpTools] = useState<any>(null);
  const [complianceReport, setComplianceReport] = useState<any>(null);
  const [securityPolicies, setSecurityPolicies] = useState<any>(null);
  const [encryptionStatus, setEncryptionStatus] = useState<any>(null);

  // ========== 安全配置管理 ==========
  const handleUpdateSecurityConfig = async () => {
    setLoading(true);
    setError(null);
    try {
      const config = {
        authentication: {
          session_timeout: 3600,
          max_failed_attempts: 5,
          password_policy: {
            min_length: 8,
            require_uppercase: true,
            require_lowercase: true,
            require_numbers: true,
            require_special_chars: true
          }
        },
        authorization: {
          default_permissions: ['read'],
          role_hierarchy: {
            admin: ['user', 'moderator'],
            moderator: ['user']
          }
        },
        audit: {
          log_level: 'detailed',
          retention_days: 90,
          real_time_alerts: true,
          compliance_standards: ['SOC2', 'ISO27001']
        }
      };

      const result = await securityEnhancedService.updateSecurityConfig(config);
      setConfigData(result);
      setSuccess('安全配置更新成功');
    } catch (err: any) {
      setError(err.message || '配置更新失败');
    } finally {
      setLoading(false);
    }
  };

  // ========== MCP工具白名单管理 ==========
  const handleUpdateMcpWhitelist = async () => {
    setLoading(true);
    setError(null);
    try {
      const whitelist = {
        allowed_tools: ['file_reader', 'web_search', 'calculator'],
        blocked_tools: ['system_admin', 'network_scanner'],
        restricted_tools: [{
          tool_name: 'database_query',
          restrictions: {
            max_executions_per_hour: 10,
            allowed_users: ['admin', 'analyst'],
            require_approval: true,
            monitoring_level: 'verbose'
          }
        }],
        global_settings: {
          default_allow: false,
          logging_enabled: true,
          alert_on_violations: true,
          auto_block_suspicious: true
        }
      };

      const result = await securityEnhancedService.updateMcpToolWhitelist(whitelist);
      setMcpTools(result);
      setSuccess('MCP工具白名单更新成功');
    } catch (err: any) {
      setError(err.message || 'MCP工具白名单更新失败');
    } finally {
      setLoading(false);
    }
  };

  // ========== 合规报告生成 ==========
  const handleGenerateComplianceReport = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = {
        compliance_standards: ['SOC2', 'ISO27001', 'GDPR'] as const,
        report_format: 'json' as const,
        time_range: {
          start_date: '2024-01-01T00:00:00Z',
          end_date: '2024-12-31T23:59:59Z'
        },
        include_sections: {
          executive_summary: true,
          control_assessments: true,
          audit_logs: true,
          risk_analysis: true,
          remediation_plans: true,
          compliance_gaps: true
        }
      };

      const result = await securityEnhancedService.generateComplianceReport(params);
      setComplianceReport(result);
      setSuccess('合规报告生成成功');
    } catch (err: any) {
      setError(err.message || '合规报告生成失败');
    } finally {
      setLoading(false);
    }
  };

  // ========== 安全策略管理 ==========
  const handleAddSecurityPolicy = async () => {
    setLoading(true);
    setError(null);
    try {
      const policy = {
        policy_name: '数据访问控制策略',
        description: '控制敏感数据的访问权限',
        policy_type: 'access_control' as const,
        priority: 'high' as const,
        scope: {
          resources: ['database', 'file_system', 'api_endpoints'],
          operations: ['read', 'write', 'delete']
        },
        rules: [{
          rule_id: 'rule_001',
          name: '管理员访问规则',
          condition: {
            field: 'user.role',
            operator: 'equals' as const,
            value: 'admin'
          },
          action: 'allow' as const
        }],
        enforcement: {
          mode: 'enforcing' as const,
          effective_date: new Date().toISOString(),
          grace_period_hours: 24
        },
        notifications: {
          on_violation: true,
          on_policy_change: true,
          notification_channels: ['email', 'slack'] as const,
          recipients: ['security@company.com']
        }
      };

      const result = await securityEnhancedService.addSecurityPolicy(policy);
      setSecurityPolicies(prev => prev ? {...prev, new_policy: result} : result);
      setSuccess('安全策略添加成功');
    } catch (err: any) {
      setError(err.message || '安全策略添加失败');
    } finally {
      setLoading(false);
    }
  };

  // ========== 通信加密 ==========
  const handleEncryptCommunication = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = {
        sender_agent_id: 'agent_001',
        recipient_agent_ids: ['agent_002', 'agent_003'],
        message: {
          content: { action: 'process_data', data: 'sensitive_information' },
          message_type: 'command' as const,
          priority: 'high' as const
        },
        encryption_settings: {
          algorithm: 'AES-256' as const,
          compression: true,
          integrity_check: true
        },
        delivery_options: {
          delivery_method: 'direct' as const,
          require_confirmation: true,
          timeout_seconds: 30
        }
      };

      const result = await securityEnhancedService.encryptCommunication(data);
      setEncryptionStatus(result);
      setSuccess('通信加密成功');
    } catch (err: any) {
      setError(err.message || '通信加密失败');
    } finally {
      setLoading(false);
    }
  };

  // ========== 加载数据 ==========
  const loadSecurityPolicies = async () => {
    try {
      const result = await securityEnhancedService.getSecurityPolicies({
        status: 'active',
        limit: 10
      });
      setSecurityPolicies(result);
    } catch (err: any) {
      setError('加载安全策略失败');
    }
  };

  useEffect(() => {
    if (activeTab === 'policies') {
      loadSecurityPolicies();
    }
  }, [activeTab]);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Shield className="h-8 w-8 text-blue-600" />
            高级安全管理
          </h1>
          <p className="text-gray-600">配置管理、工具安全、合规报告、策略控制</p>
        </div>
      </div>

      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription className="text-red-800">{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert className="border-green-200 bg-green-50">
          <CheckCircle className="h-4 w-4" />
          <AlertDescription className="text-green-800">{success}</AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="config" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            配置管理
          </TabsTrigger>
          <TabsTrigger value="mcp-tools" className="flex items-center gap-2">
            <Key className="h-4 w-4" />
            工具安全
          </TabsTrigger>
          <TabsTrigger value="compliance" className="flex items-center gap-2">
            <FileCheck className="h-4 w-4" />
            合规报告
          </TabsTrigger>
          <TabsTrigger value="policies" className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            安全策略
          </TabsTrigger>
          <TabsTrigger value="encryption" className="flex items-center gap-2">
            <Lock className="h-4 w-4" />
            通信加密
          </TabsTrigger>
          <TabsTrigger value="access" className="flex items-center gap-2">
            <Users className="h-4 w-4" />
            访问控制
          </TabsTrigger>
        </TabsList>

        <TabsContent value="config" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>安全配置管理</CardTitle>
              <CardDescription>更新系统安全配置参数</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">身份认证配置</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <label className="text-sm font-medium">会话超时(秒)</label>
                      <Input type="number" name="sessionTimeoutSeconds" defaultValue="3600" />
                    </div>
                    <div>
                      <label className="text-sm font-medium">最大失败尝试次数</label>
                      <Input type="number" name="maxFailedAttempts" defaultValue="5" />
                    </div>
                    <div>
                      <label className="text-sm font-medium">密码最小长度</label>
                      <Input type="number" name="passwordMinLength" defaultValue="8" />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">审计配置</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <label className="text-sm font-medium">日志级别</label>
                      <select name="auditLogLevel" className="w-full p-2 border border-gray-300 rounded">
                        <option value="basic">基础</option>
                        <option value="detailed" selected>详细</option>
                        <option value="verbose">详尽</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-sm font-medium">日志保留天数</label>
                      <Input type="number" name="logRetentionDays" defaultValue="90" />
                    </div>
                    <div className="flex items-center space-x-2">
                      <input type="checkbox" id="realtime-alerts" defaultChecked />
                      <label htmlFor="realtime-alerts" className="text-sm">实时告警</label>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Button onClick={handleUpdateSecurityConfig} disabled={loading} className="w-full">
                {loading ? '更新中...' : '更新安全配置'}
              </Button>

              {configData && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium mb-2">配置更新结果</h4>
                  <div className="space-y-1 text-sm">
                    <div>状态: <Badge variant={configData.success ? 'default' : 'destructive'}>
                      {configData.success ? '成功' : '失败'}
                    </Badge></div>
                    <div>更新的配置项: {configData.updated_sections?.length || 0}个</div>
                    {configData.requires_restart && (
                      <div className="text-orange-600">⚠️ 需要重启服务以生效</div>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="mcp-tools" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>MCP工具安全管理</CardTitle>
              <CardDescription>管理MCP工具的白名单、权限和使用限制</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">工具白名单</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <label className="text-sm font-medium">允许的工具</label>
                      <Textarea 
                        name="mcpAllowedTools"
                        placeholder="file_reader, web_search, calculator"
                        defaultValue="file_reader, web_search, calculator"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium">禁用的工具</label>
                      <Textarea 
                        name="mcpDisabledTools"
                        placeholder="system_admin, network_scanner"
                        defaultValue="system_admin, network_scanner"
                      />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">全局设置</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <input type="checkbox" id="default-allow" />
                      <label htmlFor="default-allow" className="text-sm">默认允许新工具</label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input type="checkbox" id="logging-enabled" defaultChecked />
                      <label htmlFor="logging-enabled" className="text-sm">启用日志记录</label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input type="checkbox" id="alert-violations" defaultChecked />
                      <label htmlFor="alert-violations" className="text-sm">违规时告警</label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input type="checkbox" id="auto-block" defaultChecked />
                      <label htmlFor="auto-block" className="text-sm">自动阻止可疑活动</label>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Button onClick={handleUpdateMcpWhitelist} disabled={loading} className="w-full">
                {loading ? '更新中...' : '更新工具白名单'}
              </Button>

              {mcpTools && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium mb-2">工具白名单更新结果</h4>
                  <div className="space-y-1 text-sm">
                    <div>状态: <Badge variant={mcpTools.status === 'updated' ? 'default' : 'destructive'}>
                      {mcpTools.status}
                    </Badge></div>
                    <div>更新的工具数量: {mcpTools.updated_tools}个</div>
                    <div>生效时间: {new Date(mcpTools.effective_date).toLocaleString()}</div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="compliance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>合规报告生成</CardTitle>
              <CardDescription>生成多标准合规报告</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="text-sm font-medium">合规标准</label>
                  <div className="space-y-2 mt-2">
                    <div className="flex items-center space-x-2">
                      <input type="checkbox" id="soc2" defaultChecked />
                      <label htmlFor="soc2" className="text-sm">SOC2</label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input type="checkbox" id="iso27001" defaultChecked />
                      <label htmlFor="iso27001" className="text-sm">ISO27001</label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input type="checkbox" id="gdpr" defaultChecked />
                      <label htmlFor="gdpr" className="text-sm">GDPR</label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input type="checkbox" id="hipaa" />
                      <label htmlFor="hipaa" className="text-sm">HIPAA</label>
                    </div>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium">报告格式</label>
                  <select name="complianceReportFormat" className="w-full p-2 border border-gray-300 rounded mt-2">
                    <option value="json" selected>JSON</option>
                    <option value="pdf">PDF</option>
                    <option value="html">HTML</option>
                    <option value="excel">Excel</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium">时间范围</label>
                  <div className="space-y-2 mt-2">
                    <Input type="date" name="complianceStartDate" defaultValue="2024-01-01" />
                    <Input type="date" name="complianceEndDate" defaultValue="2024-12-31" />
                  </div>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium">包含内容</label>
                <div className="grid grid-cols-2 gap-2 mt-2">
                  <div className="flex items-center space-x-2">
                    <input type="checkbox" id="exec-summary" defaultChecked />
                    <label htmlFor="exec-summary" className="text-sm">执行摘要</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input type="checkbox" id="control-assess" defaultChecked />
                    <label htmlFor="control-assess" className="text-sm">控制评估</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input type="checkbox" id="audit-logs" defaultChecked />
                    <label htmlFor="audit-logs" className="text-sm">审计日志</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input type="checkbox" id="risk-analysis" defaultChecked />
                    <label htmlFor="risk-analysis" className="text-sm">风险分析</label>
                  </div>
                </div>
              </div>

              <Button onClick={handleGenerateComplianceReport} disabled={loading} className="w-full">
                {loading ? '生成中...' : '生成合规报告'}
              </Button>

              {complianceReport && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium mb-2">合规报告生成结果</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>报告ID:</span>
                      <Badge>{complianceReport.report_id}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>状态:</span>
                      <Badge variant={complianceReport.status === 'completed' ? 'default' : 'secondary'}>
                        {complianceReport.status}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>总控制数:</span>
                      <span>{complianceReport.summary?.total_controls_assessed}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>关键发现:</span>
                      <span className="text-red-600">{complianceReport.summary?.critical_findings}</span>
                    </div>
                    {complianceReport.download_url && (
                      <Button variant="outline" className="w-full mt-2">
                        <Download className="h-4 w-4 mr-2" />
                        下载报告
                      </Button>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="policies" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>安全策略管理</CardTitle>
              <CardDescription>添加和管理分布式安全策略</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">策略名称</label>
                  <Input name="policyName" placeholder="数据访问控制策略" defaultValue="数据访问控制策略" />
                </div>
                <div>
                  <label className="text-sm font-medium">策略类型</label>
                  <select name="policyType" className="w-full p-2 border border-gray-300 rounded">
                    <option value="access_control" selected>访问控制</option>
                    <option value="data_protection">数据保护</option>
                    <option value="communication">通信安全</option>
                    <option value="audit">审计策略</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium">策略描述</label>
                <Textarea 
                  name="policyDescription"
                  placeholder="描述策略的目的和应用范围..."
                  defaultValue="控制敏感数据的访问权限"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">优先级</label>
                  <select name="policyPriority" className="w-full p-2 border border-gray-300 rounded">
                    <option value="low">低</option>
                    <option value="medium">中</option>
                    <option value="high" selected>高</option>
                    <option value="critical">关键</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium">执行模式</label>
                  <select name="policyEnforcement" className="w-full p-2 border border-gray-300 rounded">
                    <option value="permissive">宽松</option>
                    <option value="enforcing" selected>强制</option>
                    <option value="disabled">禁用</option>
                  </select>
                </div>
              </div>

              <Button onClick={handleAddSecurityPolicy} disabled={loading} className="w-full">
                {loading ? '添加中...' : '添加安全策略'}
              </Button>

              {securityPolicies && (
                <div className="mt-4">
                  <h4 className="font-medium mb-2">当前策略</h4>
                  <div className="space-y-2">
                    {securityPolicies.policies?.slice(0, 3).map((policy: any, idx: number) => (
                      <div key={idx} className="border rounded p-3 space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="font-medium">{policy.policy_name}</span>
                          <Badge variant={policy.status === 'active' ? 'default' : 'secondary'}>
                            {policy.status}
                          </Badge>
                        </div>
                        <div className="text-sm text-gray-600">{policy.description}</div>
                        <div className="flex justify-between text-sm">
                          <span>类型: {policy.policy_type}</span>
                          <span>优先级: {policy.priority}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="encryption" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>通信加密</CardTitle>
              <CardDescription>智能体间安全通信加密</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">发送者智能体ID</label>
                  <Input name="encryptionSender" placeholder="agent_001" defaultValue="agent_001" />
                </div>
                <div>
                  <label className="text-sm font-medium">接收者智能体ID</label>
                  <Input name="encryptionReceivers" placeholder="agent_002,agent_003" defaultValue="agent_002,agent_003" />
                </div>
              </div>

              <div>
                <label className="text-sm font-medium">消息内容</label>
                <Textarea 
                  name="encryptionMessage"
                  placeholder="输入要加密的消息内容..."
                  defaultValue='{"action": "process_data", "data": "sensitive_information"}'
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="text-sm font-medium">加密算法</label>
                  <select name="encryptionAlgorithm" className="w-full p-2 border border-gray-300 rounded">
                    <option value="AES-256" selected>AES-256</option>
                    <option value="ChaCha20">ChaCha20</option>
                    <option value="RSA-4096">RSA-4096</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium">消息类型</label>
                  <select name="encryptionMessageType" className="w-full p-2 border border-gray-300 rounded">
                    <option value="command" selected>命令</option>
                    <option value="data">数据</option>
                    <option value="status">状态</option>
                    <option value="alert">告警</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium">优先级</label>
                  <select name="encryptionPriority" className="w-full p-2 border border-gray-300 rounded">
                    <option value="low">低</option>
                    <option value="normal">正常</option>
                    <option value="high" selected>高</option>
                    <option value="urgent">紧急</option>
                  </select>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <input type="checkbox" id="compression" defaultChecked />
                  <label htmlFor="compression" className="text-sm">启用压缩</label>
                </div>
                <div className="flex items-center space-x-2">
                  <input type="checkbox" id="integrity" defaultChecked />
                  <label htmlFor="integrity" className="text-sm">完整性检查</label>
                </div>
                <div className="flex items-center space-x-2">
                  <input type="checkbox" id="confirmation" defaultChecked />
                  <label htmlFor="confirmation" className="text-sm">需要确认</label>
                </div>
              </div>

              <Button onClick={handleEncryptCommunication} disabled={loading} className="w-full">
                {loading ? '加密发送中...' : '加密并发送消息'}
              </Button>

              {encryptionStatus && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium mb-2">加密状态</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <div>消息ID: <Badge>{encryptionStatus.message_id}</Badge></div>
                      <div>加密状态: <Badge variant={encryptionStatus.encryption_status === 'success' ? 'default' : 'destructive'}>
                        {encryptionStatus.encryption_status}
                      </Badge></div>
                      <div>压缩比: {encryptionStatus.compression_ratio?.toFixed(2)}x</div>
                    </div>
                    <div>
                      <div>加密算法: {encryptionStatus.encryption_metadata?.algorithm_used}</div>
                      <div>加密耗时: {encryptionStatus.performance_metrics?.encryption_time_ms}ms</div>
                      <div>传输耗时: {encryptionStatus.performance_metrics?.transmission_time_ms}ms</div>
                    </div>
                  </div>
                  
                  {encryptionStatus.delivery_status && (
                    <div className="mt-3">
                      <h5 className="font-medium mb-1">投递状态</h5>
                      {Object.entries(encryptionStatus.delivery_status).map(([agentId, status]: [string, any]) => (
                        <div key={agentId} className="flex justify-between items-center">
                          <span>{agentId}</span>
                          <Badge variant={status.status === 'delivered' ? 'default' : status.status === 'pending' ? 'secondary' : 'destructive'}>
                            {status.status}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="access" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>访问权限撤销</CardTitle>
              <CardDescription>撤销智能体的访问权限</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">目标智能体ID</label>
                  <Input name="revokeAgentId" placeholder="输入要撤销权限的智能体ID..." />
                </div>
                <div>
                  <label className="text-sm font-medium">撤销原因</label>
                  <select name="revokeReason" className="w-full p-2 border border-gray-300 rounded">
                    <option value="security_breach">安全漏洞</option>
                    <option value="policy_violation">策略违规</option>
                    <option value="role_change">角色变更</option>
                    <option value="termination">终止服务</option>
                    <option value="maintenance">维护</option>
                    <option value="other">其他</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium">撤销描述</label>
                <Textarea name="revokeDescription" placeholder="详细描述撤销权限的原因和背景..." />
              </div>

              <div className="space-y-3">
                <label className="text-sm font-medium">撤销范围</label>
                <div className="flex items-center space-x-2">
                  <input type="checkbox" id="revoke-all" defaultChecked />
                  <label htmlFor="revoke-all" className="text-sm">撤销所有权限</label>
                </div>
                <div className="flex items-center space-x-2">
                  <input type="checkbox" id="terminate-sessions" defaultChecked />
                  <label htmlFor="terminate-sessions" className="text-sm">终止活跃会话</label>
                </div>
                <div className="flex items-center space-x-2">
                  <input type="checkbox" id="clear-cache" defaultChecked />
                  <label htmlFor="clear-cache" className="text-sm">清除缓存权限</label>
                </div>
                <div className="flex items-center space-x-2">
                  <input type="checkbox" id="backup-state" defaultChecked />
                  <label htmlFor="backup-state" className="text-sm">备份智能体状态</label>
                </div>
              </div>

              <div className="space-y-3">
                <label className="text-sm font-medium">通知设置</label>
                <div className="flex items-center space-x-2">
                  <input type="checkbox" id="notify-agent" />
                  <label htmlFor="notify-agent" className="text-sm">通知目标智能体</label>
                </div>
                <div className="flex items-center space-x-2">
                  <input type="checkbox" id="notify-admin" defaultChecked />
                  <label htmlFor="notify-admin" className="text-sm">通知管理员</label>
                </div>
                <div className="flex items-center space-x-2">
                  <input type="checkbox" id="notify-users" />
                  <label htmlFor="notify-users" className="text-sm">通知受影响用户</label>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <input type="checkbox" id="immediate-action" defaultChecked />
                <label htmlFor="immediate-action" className="text-sm font-medium">立即执行撤销操作</label>
              </div>

              <Button disabled={loading} className="w-full bg-red-600 hover:bg-red-700">
                {loading ? '撤销中...' : '撤销访问权限'}
              </Button>

              <Alert className="border-red-200 bg-red-50">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription className="text-red-800">
                  撤销访问权限是不可逆操作，请确认所有信息正确后再执行。
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SecurityManagementEnhancedPage;
