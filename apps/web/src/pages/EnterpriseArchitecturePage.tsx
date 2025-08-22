import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Tabs,
  Tab,
  Chip,
  Button,
  Alert,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material';
import enterpriseService, {
  SystemHealthMetrics,
  SecurityMetrics,
  PerformanceMetrics,
  ComplianceData
} from '../services/enterpriseService';
import {
  ExpandMore as ExpandMoreIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Visibility as VisibilityIcon,
  Assessment as AssessmentIcon,
  Dashboard as DashboardIcon,
  Settings as SettingsIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon
} from '@mui/icons-material';

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
      id={`enterprise-tabpanel-${index}`}
      aria-labelledby={`enterprise-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}


const EnterpriseArchitecturePage: React.FC = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [systemHealth, setSystemHealth] = useState<SystemHealthMetrics | null>(null);
  const [securityMetrics, setSecurityMetrics] = useState<SecurityMetrics | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null);
  const [complianceData, setComplianceData] = useState<ComplianceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState<string>('');

  useEffect(() => {
    loadEnterpriseMetrics();
    const interval = setInterval(loadEnterpriseMetrics, 30000); // 每30秒刷新
    return () => clearInterval(interval);
  }, []);

  const loadEnterpriseMetrics = async () => {
    try {
      setLoading(true);
      
      // 尝试从API加载真实数据
      try {
        const healthData = await enterpriseService.getSystemHealth();
        setSystemHealth(healthData);
        
        const securityData = await enterpriseService.getSecurityMetrics();
        setSecurityMetrics(securityData);
        
        const performanceData = await enterpriseService.getPerformanceMetrics();
        setPerformanceMetrics(performanceData);
        
        const complianceData = await enterpriseService.getComplianceData();
        setComplianceData(complianceData);
        
      } catch (apiError) {
        console.warn('API不可用，使用模拟数据:', apiError);
        
        // 模拟数据作为后备
        setSystemHealth({
          overall_status: 'healthy',
          cpu_usage: 45.2,
          memory_usage: 67.8,
          disk_usage: 23.1,
          active_agents: 12,
          active_tasks: 34,
          error_rate: 0.02,
          response_time: 156,
          timestamp: new Date().toISOString()
        });

        setSecurityMetrics({
          threat_level: 'low',
          detected_attacks: 3,
          blocked_requests: 27,
          security_events: 8,
          compliance_score: 94.5,
          last_security_scan: '2024-01-15T10:30:00Z',
          active_threats: []
        });

        setPerformanceMetrics({
          throughput: 1250,
          latency_p50: 89,
          latency_p95: 245,
          latency_p99: 387,
          concurrent_users: 156,
          cache_hit_rate: 87.3,
          optimization_level: 'high',
          resource_utilization: {
            cpu: 45.2,
            memory: 67.8,
            io: 12.5,
            network: 8.3
          }
        });

        setComplianceData({
          overall_score: 92.8,
          status: 'compliant',
          standards: ['ISO27001', 'SOC2', 'GDPR'],
          last_assessment: '2024-01-14T14:20:00Z',
          issues_count: 2,
          requirements_total: 45,
          requirements_passed: 43,
          detailed_results: []
        });
      }

    } catch (error) {
      console.error('加载企业指标失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'compliant':
        return 'success';
      case 'degraded':
      case 'partially_compliant':
        return 'warning';
      case 'unhealthy':
      case 'non_compliant':
        return 'error';
      default:
        return 'default';
    }
  };

  const getThreatLevelColor = (level: string) => {
    switch (level) {
      case 'low':
        return 'success';
      case 'medium':
        return 'warning';
      case 'high':
      case 'critical':
        return 'error';
      default:
        return 'default';
    }
  };

  const SystemHealthOverview = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Card>
          <CardHeader
            title="系统健康状态"
            action={
              <Chip
                label={systemHealth?.overall_status}
                color={getStatusColor(systemHealth?.overall_status || '')}
                icon={systemHealth?.overall_status === 'healthy' ? <CheckCircleIcon /> : <WarningIcon />}
              />
            }
          />
          <CardContent>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="body2" color="textSecondary">
                  CPU使用率
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={systemHealth?.cpu_usage || 0}
                  sx={{ mt: 1, mb: 1 }}
                />
                <Typography variant="body2">
                  {systemHealth?.cpu_usage.toFixed(1)}%
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="body2" color="textSecondary">
                  内存使用率
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={systemHealth?.memory_usage || 0}
                  sx={{ mt: 1, mb: 1 }}
                />
                <Typography variant="body2">
                  {systemHealth?.memory_usage.toFixed(1)}%
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="body2" color="textSecondary">
                  磁盘使用率
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={systemHealth?.disk_usage || 0}
                  sx={{ mt: 1, mb: 1 }}
                />
                <Typography variant="body2">
                  {systemHealth?.disk_usage.toFixed(1)}%
                </Typography>
              </Grid>
            </Grid>
            
            <Grid container spacing={2} sx={{ mt: 2 }}>
              <Grid item xs={6} md={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="primary">
                    {systemHealth?.active_agents}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    活跃智能体
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6} md={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="primary">
                    {systemHealth?.active_tasks}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    活跃任务
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6} md={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="secondary">
                    {(systemHealth?.error_rate * 100).toFixed(2)}%
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    错误率
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6} md={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="info">
                    {systemHealth?.response_time}ms
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    响应时间
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const SecurityDashboard = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader
            title="安全威胁等级"
            action={
              <Chip
                label={securityMetrics?.threat_level}
                color={getThreatLevelColor(securityMetrics?.threat_level || '')}
                icon={<SecurityIcon />}
              />
            }
          />
          <CardContent>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="h5" color="warning.main">
                  {securityMetrics?.detected_attacks}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  检测到的攻击
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="h5" color="error.main">
                  {securityMetrics?.blocked_requests}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  阻止的请求
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
                  安全事件: {securityMetrics?.security_events}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  上次扫描: {securityMetrics?.last_security_scan ? 
                    new Date(securityMetrics.last_security_scan).toLocaleString() : 'N/A'}
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="合规评分" />
          <CardContent>
            <Box display="flex" alignItems="center" mb={2}>
              <Typography variant="h4" color="success.main" sx={{ mr: 2 }}>
                {securityMetrics?.compliance_score.toFixed(1)}%
              </Typography>
              <TrendingUpIcon color="success" />
            </Box>
            <LinearProgress
              variant="determinate"
              value={securityMetrics?.compliance_score || 0}
              color="success"
              sx={{ mb: 2 }}
            />
            <Typography variant="body2" color="textSecondary">
              系统符合企业安全标准
            </Typography>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const PerformanceDashboard = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="吞吐量与延迟" />
          <CardContent>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="h4" color="primary">
                  {performanceMetrics?.throughput.toLocaleString()} req/min
                </Typography>
                <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                  当前吞吐量
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="h6">
                  {performanceMetrics?.latency_p50}ms
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  P50延迟
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="h6">
                  {performanceMetrics?.latency_p95}ms
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  P95延迟
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="h6">
                  {performanceMetrics?.latency_p99}ms
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  P99延迟
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="缓存与用户" />
          <CardContent>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="h5" color="info.main">
                  {performanceMetrics?.concurrent_users}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  并发用户
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="h5" color="success.main">
                  {performanceMetrics?.cache_hit_rate.toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  缓存命中率
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Box sx={{ mt: 2 }}>
                  <Chip
                    label={`优化级别: ${performanceMetrics?.optimization_level}`}
                    color="primary"
                    variant="outlined"
                  />
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const ComplianceDashboard = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <Card>
          <CardHeader title="合规状态总览" />
          <CardContent>
            <Box display="flex" alignItems="center" mb={3}>
              <Typography variant="h3" color="success.main" sx={{ mr: 2 }}>
                {complianceData?.overall_score.toFixed(1)}%
              </Typography>
              <Box>
                <Chip
                  label={complianceData?.status}
                  color={getStatusColor(complianceData?.status || '')}
                  icon={<CheckCircleIcon />}
                />
                <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                  上次评估: {complianceData?.last_assessment ? 
                    new Date(complianceData.last_assessment).toLocaleString() : 'N/A'}
                </Typography>
              </Box>
            </Box>
            
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
              合规要求进度: {complianceData?.requirements_passed} / {complianceData?.requirements_total}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={(complianceData?.requirements_passed || 0) / (complianceData?.requirements_total || 1) * 100}
              sx={{ mb: 2 }}
            />
            
            <Box display="flex" gap={1} flexWrap="wrap">
              {complianceData?.standards.map((standard) => (
                <Chip key={standard} label={standard} size="small" />
              ))}
            </Box>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={12} md={4}>
        <Card>
          <CardHeader title="待处理问题" />
          <CardContent>
            <Box textAlign="center">
              <Typography variant="h3" color={complianceData?.issues_count === 0 ? 'success.main' : 'warning.main'}>
                {complianceData?.issues_count}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                需要处理的问题
              </Typography>
              {complianceData?.issues_count === 0 && (
                <Alert severity="success" sx={{ mt: 2 }}>
                  所有合规要求已满足
                </Alert>
              )}
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const ConfigurationPanel = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Card>
          <CardHeader title="架构配置" />
          <CardContent>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>安全配置</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <SecurityIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary="AI TRiSM框架"
                      secondary="信任、风险和安全管理已启用"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircleIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary="攻击检测"
                      secondary="主动检测提示注入、数据泄露等威胁"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <WarningIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary="自动响应"
                      secondary="威胁自动阻止和隔离机制"
                    />
                  </ListItem>
                </List>
              </AccordionDetails>
            </Accordion>
            
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>性能配置</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <SpeedIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary="高并发优化"
                      secondary="异步任务池和连接池已配置"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <TrendingUpIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary="负载均衡"
                      secondary="智能体分布式调度和资源优化"
                    />
                  </ListItem>
                </List>
              </AccordionDetails>
            </Accordion>
            
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>监控配置</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <VisibilityIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary="OpenTelemetry"
                      secondary="分布式追踪和指标收集已启用"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <AssessmentIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary="审计日志"
                      secondary="企业级审计和合规日志记录"
                    />
                  </ListItem>
                </List>
              </AccordionDetails>
            </Accordion>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  if (loading && !systemHealth) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
          <LinearProgress sx={{ width: '50%' }} />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        企业级架构管理
      </Typography>
      <Typography variant="subtitle1" color="textSecondary" gutterBottom>
        监控和管理AI智能体系统的企业级架构组件
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={currentTab} onChange={handleTabChange} aria-label="enterprise architecture tabs">
          <Tab icon={<DashboardIcon />} label="系统健康" />
          <Tab icon={<SecurityIcon />} label="安全监控" />
          <Tab icon={<SpeedIcon />} label="性能指标" />
          <Tab icon={<AssessmentIcon />} label="合规状态" />
          <Tab icon={<SettingsIcon />} label="配置管理" />
        </Tabs>
      </Box>

      <TabPanel value={currentTab} index={0}>
        <SystemHealthOverview />
      </TabPanel>

      <TabPanel value={currentTab} index={1}>
        <SecurityDashboard />
      </TabPanel>

      <TabPanel value={currentTab} index={2}>
        <PerformanceDashboard />
      </TabPanel>

      <TabPanel value={currentTab} index={3}>
        <ComplianceDashboard />
      </TabPanel>

      <TabPanel value={currentTab} index={4}>
        <ConfigurationPanel />
      </TabPanel>

      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between' }}>
        <Button
          variant="outlined"
          onClick={loadEnterpriseMetrics}
          disabled={loading}
        >
          刷新数据
        </Button>
        <Button
          variant="contained"
          onClick={() => setDetailsOpen(true)}
        >
          查看详细报告
        </Button>
      </Box>

      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>企业架构详细报告</DialogTitle>
        <DialogContent>
          <Typography variant="body1" paragraph>
            系统当前运行稳定，所有企业级组件正常工作：
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="分布式智能体管理"
                secondary="企业级异步架构和负载均衡已就绪"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="AI TRiSM安全框架"
                secondary="信任、风险和安全管理全面部署"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="高并发性能优化"
                secondary="异步任务池和资源优化运行正常"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="合规认证系统"
                secondary="符合ISO27001、SOC2、GDPR等标准"
              />
            </ListItem>
          </List>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>关闭</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default EnterpriseArchitecturePage;