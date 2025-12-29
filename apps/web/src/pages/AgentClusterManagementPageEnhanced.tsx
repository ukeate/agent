import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Tabs,
  Tab,
  Card,
  CardContent,
  Grid,
  Button,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  CircularProgress,
  Tooltip
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  CloudQueue as CloudIcon,
  Speed as SpeedIcon,
  Security as SecurityIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  Assessment as AssessmentIcon,
  AutoFixHigh as AutoIcon,
  Group as GroupIcon
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { clusterManagementServiceEnhanced } from '../services/clusterManagementServiceEnhanced';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`cluster-tabpanel-${index}`}
      aria-labelledby={`cluster-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export default function AgentClusterManagementPageEnhanced() {
  const [currentTab, setCurrentTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadBalancingStrategies, setLoadBalancingStrategies] = useState<any[]>([]);
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [performanceProfiles, setPerformanceProfiles] = useState<any[]>([]);
  const [capacityForecast, setCapacityForecast] = useState<any>(null);
  const [securityAudits, setSecurityAudits] = useState<any[]>([]);
  const [anomalies, setAnomalies] = useState<any[]>([]);
  const [workflows, setWorkflows] = useState<any[]>([]);
  const [reports, setReports] = useState<any[]>([]);
  
  const [strategyDialog, setStrategyDialog] = useState(false);
  const [auditDialog, setAuditDialog] = useState(false);
  const [workflowDialog, setWorkflowDialog] = useState(false);
  const [strategyName, setStrategyName] = useState('');
  const [strategyAlgorithm, setStrategyAlgorithm] = useState('round_robin');
  const [auditSelections, setAuditSelections] = useState({
    accessControl: false,
    networkSecurity: false,
    dataProtection: false,
    compliance: false,
  });
  const [workflowName, setWorkflowName] = useState('');
  const [workflowType, setWorkflowType] = useState('scaling');

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const loadLoadBalancingData = async () => {
    try {
      setLoading(true);
      const strategies = await clusterManagementServiceEnhanced.getLoadBalancingStrategies();
      setLoadBalancingStrategies(strategies);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadHealthData = async () => {
    try {
      setLoading(true);
      const status = await clusterManagementServiceEnhanced.getDeepHealthAnalysis();
      setHealthStatus(status);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadPerformanceData = async () => {
    try {
      setLoading(true);
      const profiles = await clusterManagementServiceEnhanced.getPerformanceProfiles();
      setPerformanceProfiles(profiles);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadCapacityData = async () => {
    try {
      setLoading(true);
      const forecast = await clusterManagementServiceEnhanced.generateCapacityForecast({
        forecast_horizon_days: 30,
        scenarios: ['conservative', 'moderate', 'aggressive'],
        include_recommendations: true
      });
      setCapacityForecast(forecast);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadSecurityData = async () => {
    try {
      setLoading(true);
      const audits = await clusterManagementServiceEnhanced.getSecurityAudits();
      setSecurityAudits(audits);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadAnomalyData = async () => {
    try {
      setLoading(true);
      const anomalyData = await clusterManagementServiceEnhanced.detectAnomalies({
        detection_window_hours: 24,
        sensitivity: 'medium',
        include_predictions: true
      });
      setAnomalies(anomalyData.anomalies);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadWorkflowData = async () => {
    try {
      setLoading(true);
      const workflowList = await clusterManagementServiceEnhanced.getAutomatedWorkflows();
      setWorkflows(workflowList);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadReportData = async () => {
    try {
      setLoading(true);
      const reportList = await clusterManagementServiceEnhanced.getComprehensiveReports();
      setReports(reportList);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    switch (currentTab) {
      case 0:
        loadLoadBalancingData();
        break;
      case 1:
        loadHealthData();
        break;
      case 2:
        loadPerformanceData();
        break;
      case 3:
        loadCapacityData();
        break;
      case 4:
        loadSecurityData();
        break;
      case 5:
        loadAnomalyData();
        break;
      case 6:
        loadWorkflowData();
        break;
      case 7:
        loadReportData();
        break;
    }
  }, [currentTab]);

  const createLoadBalancingStrategy = async (strategyData: any) => {
    try {
      setError(null);
      setLoading(true);
      await clusterManagementServiceEnhanced.createLoadBalancingStrategy(strategyData);
      setStrategyDialog(false);
      setStrategyName('');
      setStrategyAlgorithm('round_robin');
      loadLoadBalancingData();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const triggerSecurityAudit = async (auditConfig: any) => {
    try {
      setError(null);
      setLoading(true);
      await clusterManagementServiceEnhanced.triggerSecurityAudit(auditConfig);
      setAuditDialog(false);
      setAuditSelections({
        accessControl: false,
        networkSecurity: false,
        dataProtection: false,
        compliance: false,
      });
      loadSecurityData();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const createWorkflow = async (workflowData: any) => {
    try {
      setError(null);
      setLoading(true);
      await clusterManagementServiceEnhanced.createAutomationWorkflow(workflowData);
      setWorkflowDialog(false);
      setWorkflowName('');
      setWorkflowType('scaling');
      loadWorkflowData();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 2, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        智能集群管理平台
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={currentTab} onChange={handleTabChange} aria-label="cluster management tabs" variant="scrollable">
          <Tab icon={<CloudIcon />} label="智能负载均衡" />
          <Tab icon={<SpeedIcon />} label="深度健康监控" />
          <Tab icon={<AnalyticsIcon />} label="性能分析优化" />
          <Tab icon={<AssessmentIcon />} label="容量预测规划" />
          <Tab icon={<SecurityIcon />} label="安全审计合规" />
          <Tab icon={<AutoIcon />} label="异常检测预警" />
          <Tab icon={<SettingsIcon />} label="自动化工作流" />
          <Tab icon={<GroupIcon />} label="综合报告中心" />
        </Tabs>
      </Box>

      <TabPanel value={currentTab} index={0}>
        <Card>
          <CardContent>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">智能负载均衡策略</Typography>
              <Button 
                variant="contained" 
                onClick={() => setStrategyDialog(true)}
                disabled={loading}
              >
                创建新策略
              </Button>
            </Box>
            
            {loading ? <CircularProgress /> : (
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>策略名称</TableCell>
                      <TableCell>算法类型</TableCell>
                      <TableCell>健康检查</TableCell>
                      <TableCell>故障转移</TableCell>
                      <TableCell>性能改进</TableCell>
                      <TableCell>状态</TableCell>
                      <TableCell>操作</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {loadBalancingStrategies.map((strategy) => (
                      <TableRow key={strategy.strategy_id}>
                        <TableCell>{strategy.name}</TableCell>
                        <TableCell>
                          <Chip label={strategy.algorithm} size="small" />
                        </TableCell>
                        <TableCell>
                          {strategy.health_check_settings?.interval_seconds}s 间隔
                        </TableCell>
                        <TableCell>
                          {strategy.failover_settings?.enable_automatic_failover ? '已启用' : '已禁用'}
                        </TableCell>
                        <TableCell>
                          {strategy.estimated_performance_improvement}%
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={strategy.is_active ? '运行中' : '停止'} 
                            color={strategy.is_active ? 'success' : 'default'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Button size="small">配置</Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </CardContent>
        </Card>
      </TabPanel>

      <TabPanel value={currentTab} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>集群健康概览</Typography>
                {healthStatus && (
                  <Box>
                    <Box display="flex" alignItems="center" mb={2}>
                      <Typography variant="body1">整体健康评分: </Typography>
                      <Chip 
                        label={`${healthStatus.overall_health_score}/100`}
                        color={healthStatus.overall_health_score > 80 ? 'success' : healthStatus.overall_health_score > 60 ? 'warning' : 'error'}
                        sx={{ ml: 1 }}
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {healthStatus.health_summary}
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>预测性维护</Typography>
                {healthStatus?.predictive_maintenance && (
                  <List>
                    {healthStatus.predictive_maintenance.upcoming_maintenance.map((item: any, index: number) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <AutoIcon color={item.urgency === 'high' ? 'error' : 'warning'} />
                        </ListItemIcon>
                        <ListItemText 
                          primary={item.description}
                          secondary={`预计时间: ${item.estimated_time}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>性能趋势分析</Typography>
                {healthStatus?.performance_trends && (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={healthStatus.performance_trends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <RechartsTooltip />
                      <Line type="monotone" dataKey="cpu_trend" stroke="#8884d8" name="CPU趋势" />
                      <Line type="monotone" dataKey="memory_trend" stroke="#82ca9d" name="内存趋势" />
                      <Line type="monotone" dataKey="response_time_trend" stroke="#ffc658" name="响应时间趋势" />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={currentTab} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>性能档案分析</Typography>
                {performanceProfiles.length > 0 && (
                  <TableContainer component={Paper}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>智能体ID</TableCell>
                          <TableCell>整体评分</TableCell>
                          <TableCell>瓶颈分析</TableCell>
                          <TableCell>优化建议数</TableCell>
                          <TableCell>预期改进</TableCell>
                          <TableCell>操作</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {performanceProfiles.map((profile) => (
                          <TableRow key={profile.agent_id}>
                            <TableCell>{profile.agent_id}</TableCell>
                            <TableCell>
                              <Chip 
                                label={profile.overall_performance_score} 
                                color={profile.overall_performance_score > 80 ? 'success' : profile.overall_performance_score > 60 ? 'warning' : 'error'}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>
                              {profile.bottlenecks?.map((bottleneck: string) => (
                                <Chip key={bottleneck} label={bottleneck} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
                              ))}
                            </TableCell>
                            <TableCell>{profile.optimization_recommendations?.length || 0}</TableCell>
                            <TableCell>{profile.expected_improvement}%</TableCell>
                            <TableCell>
                              <Button size="small">查看详情</Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={currentTab} index={3}>
        <Grid container spacing={3}>
          {capacityForecast && (
            <>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>容量预测</Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={capacityForecast.forecast_data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <RechartsTooltip />
                        <Line type="monotone" dataKey="conservative" stroke="#8884d8" name="保守预测" />
                        <Line type="monotone" dataKey="moderate" stroke="#82ca9d" name="适中预测" />
                        <Line type="monotone" dataKey="aggressive" stroke="#ffc658" name="激进预测" />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>资源建议</Typography>
                    <List>
                      {capacityForecast.recommendations?.resource_scaling_suggestions.map((suggestion: string, index: number) => (
                        <ListItem key={index}>
                          <ListItemText primary={suggestion} />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>成本优化分析</Typography>
                    {capacityForecast.cost_analysis && (
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={4}>
                          <Box textAlign="center">
                            <Typography variant="h4" color="primary">
                              ${capacityForecast.cost_analysis.current_monthly_cost}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">当前月成本</Typography>
                          </Box>
                        </Grid>
                        <Grid item xs={12} md={4}>
                          <Box textAlign="center">
                            <Typography variant="h4" color="warning.main">
                              ${capacityForecast.cost_analysis.projected_monthly_cost}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">预测月成本</Typography>
                          </Box>
                        </Grid>
                        <Grid item xs={12} md={4}>
                          <Box textAlign="center">
                            <Typography variant="h4" color="success.main">
                              ${capacityForecast.cost_analysis.potential_savings}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">潜在节省</Typography>
                          </Box>
                        </Grid>
                      </Grid>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </>
          )}
        </Grid>
      </TabPanel>

      <TabPanel value={currentTab} index={4}>
        <Card>
          <CardContent>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">安全审计报告</Typography>
              <Button 
                variant="contained" 
                color="secondary"
                onClick={() => setAuditDialog(true)}
                disabled={loading}
              >
                启动安全审计
              </Button>
            </Box>
            
            {loading ? <CircularProgress /> : (
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>审计ID</TableCell>
                      <TableCell>审计类型</TableCell>
                      <TableCell>风险评分</TableCell>
                      <TableCell>发现问题</TableCell>
                      <TableCell>合规状态</TableCell>
                      <TableCell>创建时间</TableCell>
                      <TableCell>操作</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {securityAudits.map((audit) => (
                      <TableRow key={audit.audit_id}>
                        <TableCell>{audit.audit_id}</TableCell>
                        <TableCell>
                          {audit.audit_types?.map((type: string) => (
                            <Chip key={type} label={type} size="small" sx={{ mr: 0.5 }} />
                          ))}
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={audit.overall_risk_score}
                            color={audit.overall_risk_score < 30 ? 'success' : audit.overall_risk_score < 70 ? 'warning' : 'error'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{audit.findings?.length || 0}</TableCell>
                        <TableCell>
                          <Chip 
                            label={audit.compliance_status}
                            color={audit.compliance_status === 'compliant' ? 'success' : 'warning'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{new Date(audit.created_at).toLocaleString()}</TableCell>
                        <TableCell>
                          <Button size="small">查看报告</Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </CardContent>
        </Card>
      </TabPanel>

      <TabPanel value={currentTab} index={5}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>异常检测结果</Typography>
                {anomalies.length > 0 ? (
                  <TableContainer component={Paper}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>异常类型</TableCell>
                          <TableCell>严重程度</TableCell>
                          <TableCell>置信度</TableCell>
                          <TableCell>影响智能体</TableCell>
                          <TableCell>描述</TableCell>
                          <TableCell>检测时间</TableCell>
                          <TableCell>操作</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {anomalies.map((anomaly, index) => (
                          <TableRow key={index}>
                            <TableCell>
                              <Chip label={anomaly.anomaly_type} size="small" />
                            </TableCell>
                            <TableCell>
                              <Chip 
                                label={anomaly.severity}
                                color={anomaly.severity === 'high' ? 'error' : anomaly.severity === 'medium' ? 'warning' : 'default'}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>{(anomaly.confidence * 100).toFixed(1)}%</TableCell>
                            <TableCell>{anomaly.affected_agents?.join(', ')}</TableCell>
                            <TableCell>{anomaly.description}</TableCell>
                            <TableCell>{new Date(anomaly.detected_at).toLocaleString()}</TableCell>
                            <TableCell>
                              <Button size="small">调查</Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                ) : (
                  <Alert severity="success">未检测到异常，系统运行正常</Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={currentTab} index={6}>
        <Card>
          <CardContent>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">自动化工作流</Typography>
              <Button 
                variant="contained" 
                onClick={() => setWorkflowDialog(true)}
                disabled={loading}
              >
                创建工作流
              </Button>
            </Box>
            
            {workflows.length > 0 ? (
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>工作流名称</TableCell>
                      <TableCell>类型</TableCell>
                      <TableCell>触发器</TableCell>
                      <TableCell>状态</TableCell>
                      <TableCell>执行次数</TableCell>
                      <TableCell>成功率</TableCell>
                      <TableCell>操作</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {workflows.map((workflow) => (
                      <TableRow key={workflow.workflow_id}>
                        <TableCell>{workflow.name}</TableCell>
                        <TableCell>
                          <Chip label={workflow.workflow_type} size="small" />
                        </TableCell>
                        <TableCell>{workflow.trigger_type}</TableCell>
                        <TableCell>
                          <Chip 
                            label={workflow.is_enabled ? '启用' : '禁用'}
                            color={workflow.is_enabled ? 'success' : 'default'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{workflow.execution_count}</TableCell>
                        <TableCell>{(workflow.success_rate * 100).toFixed(1)}%</TableCell>
                        <TableCell>
                          <Button size="small">编辑</Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Alert severity="info">暂无工作流配置</Alert>
            )}
          </CardContent>
        </Card>
      </TabPanel>

      <TabPanel value={currentTab} index={7}>
        <Grid container spacing={3}>
          {reports.map((report) => (
            <Grid item xs={12} md={6} key={report.report_id}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>{report.name}</Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {report.description}
                  </Typography>
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Chip 
                      label={report.report_type} 
                      size="small" 
                      color="primary"
                    />
                    <Button size="small" variant="outlined">
                      查看报告
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </TabPanel>

      {/* 创建负载均衡策略对话框 */}
      <Dialog open={strategyDialog} onClose={() => setStrategyDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>创建负载均衡策略</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="策略名称"
                  variant="outlined"
                  value={strategyName}
                  onChange={(event) => setStrategyName(event.target.value)}
                />
              </Grid>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>负载均衡算法</InputLabel>
                  <Select
                    value={strategyAlgorithm}
                    onChange={(event) => setStrategyAlgorithm(event.target.value)}
                  >
                    <MenuItem value="round_robin">轮询</MenuItem>
                    <MenuItem value="least_connections">最少连接</MenuItem>
                    <MenuItem value="weighted_round_robin">加权轮询</MenuItem>
                    <MenuItem value="least_response_time">最短响应时间</MenuItem>
                    <MenuItem value="adaptive">自适应</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStrategyDialog(false)}>取消</Button>
          <Button
            variant="contained"
            disabled={loading || !strategyName.trim()}
            onClick={() => {
              const name = strategyName.trim();
              if (!name) {
                setError('请输入策略名称');
                return;
              }
              createLoadBalancingStrategy({
                name,
                algorithm: strategyAlgorithm,
                health_check_settings: {
                  interval_seconds: 30,
                  timeout_seconds: 5,
                  failure_threshold: 3,
                  success_threshold: 2,
                },
                failover_settings: {
                  enable_automatic_failover: true,
                  failover_delay_seconds: 10,
                  backup_agent_count: 1,
                },
              });
            }}
          >
            创建
          </Button>
        </DialogActions>
      </Dialog>

      {/* 安全审计对话框 */}
      <Dialog open={auditDialog} onClose={() => setAuditDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>启动安全审计</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="body1" gutterBottom>选择审计类型：</Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={auditSelections.accessControl}
                      onChange={(event) => setAuditSelections((prev) => ({ ...prev, accessControl: event.target.checked }))}
                    />
                  }
                  label="访问控制审计"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={auditSelections.networkSecurity}
                      onChange={(event) => setAuditSelections((prev) => ({ ...prev, networkSecurity: event.target.checked }))}
                    />
                  }
                  label="网络安全审计"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={auditSelections.dataProtection}
                      onChange={(event) => setAuditSelections((prev) => ({ ...prev, dataProtection: event.target.checked }))}
                    />
                  }
                  label="数据保护审计"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={auditSelections.compliance}
                      onChange={(event) => setAuditSelections((prev) => ({ ...prev, compliance: event.target.checked }))}
                    />
                  }
                  label="合规性检查"
                />
              </Grid>
            </Grid>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAuditDialog(false)}>取消</Button>
          <Button
            variant="contained"
            color="secondary"
            disabled={
              loading ||
              (!auditSelections.accessControl &&
                !auditSelections.networkSecurity &&
                !auditSelections.dataProtection &&
                !auditSelections.compliance)
            }
            onClick={() => {
              const securityDomains: string[] = [];
              if (auditSelections.accessControl) securityDomains.push('access_control');
              if (auditSelections.networkSecurity) securityDomains.push('network_security');
              if (auditSelections.dataProtection) securityDomains.push('data_protection');
              const auditFrameworks = auditSelections.compliance ? ['compliance'] : [];
              if (securityDomains.length === 0 && auditFrameworks.length === 0) {
                setError('请至少选择一种审计类型');
                return;
              }
              triggerSecurityAudit({
                audit_frameworks: auditFrameworks,
                security_domains: securityDomains,
              });
            }}
          >
            启动审计
          </Button>
        </DialogActions>
      </Dialog>

      {/* 创建工作流对话框 */}
      <Dialog open={workflowDialog} onClose={() => setWorkflowDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>创建自动化工作流</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="工作流名称"
                  variant="outlined"
                  value={workflowName}
                  onChange={(event) => setWorkflowName(event.target.value)}
                />
              </Grid>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>工作流类型</InputLabel>
                  <Select
                    value={workflowType}
                    onChange={(event) => setWorkflowType(event.target.value)}
                  >
                    <MenuItem value="scaling">自动扩缩容</MenuItem>
                    <MenuItem value="healing">自动修复</MenuItem>
                    <MenuItem value="optimization">性能优化</MenuItem>
                    <MenuItem value="maintenance">预防性维护</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setWorkflowDialog(false)}>取消</Button>
          <Button
            variant="contained"
            disabled={loading || !workflowName.trim()}
            onClick={() => {
              const name = workflowName.trim();
              if (!name) {
                setError('请输入工作流名称');
                return;
              }
              createWorkflow({
                name,
                workflow_type: workflowType,
                trigger_type: 'manual',
                is_enabled: true,
              });
            }}
          >
            创建
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
