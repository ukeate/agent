/**
 * 实验实时监控仪表板
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Chip,
  Stack,
  LinearProgress,
  Alert,
  Button,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Badge,
  Avatar,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemAvatar,
  CircularProgress,
  Tab,
  Tabs,
  TextField,
  InputAdornment
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  FullscreenExit as FullscreenExitIcon,
  Fullscreen as FullscreenIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Speed as SpeedIcon,
  Science as ScienceIcon,
  Group as GroupIcon,
  Timeline as TimelineIcon,
  NotificationsActive as NotificationsIcon,
  Assessment as AssessmentIcon,
  BugReport as BugIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Settings as SettingsIcon,
  Download as DownloadIcon,
  DateRange as DateRangeIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
  RadialBarChart,
  RadialBar,
  Scatter,
  ScatterChart,
  ComposedChart
} from 'recharts';
import { format, subHours, subDays } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import { experimentService } from '../services/experimentService';
import { metricsService } from '../services/metricsService';

// 时间范围选项
const TIME_RANGES = [
  { value: '1h', label: '1小时' },
  { value: '6h', label: '6小时' },
  { value: '24h', label: '24小时' },
  { value: '7d', label: '7天' },
  { value: '30d', label: '30天' }
];

// 刷新间隔选项
const REFRESH_INTERVALS = [
  { value: 0, label: '不刷新' },
  { value: 5, label: '5秒' },
  { value: 10, label: '10秒' },
  { value: 30, label: '30秒' },
  { value: 60, label: '1分钟' }
];

// 图表颜色
const COLORS = {
  primary: '#3f51b5',
  success: '#4caf50',
  warning: '#ff9800',
  error: '#f44336',
  info: '#2196f3',
  control: '#9e9e9e',
  variant1: '#673ab7',
  variant2: '#009688',
  variant3: '#ff5722'
};

interface DashboardData {
  experiment: any;
  metrics: any[];
  events: any[];
  alerts: any[];
  health: {
    status: 'healthy' | 'warning' | 'error';
    issues: string[];
  };
  realTimeStats: {
    currentUsers: number;
    requestsPerSecond: number;
    errorRate: number;
    avgLatency: number;
  };
}

const ExperimentDashboardPage: React.FC = () => {
  const [selectedExperiment, setSelectedExperiment] = useState<string>('');
  const [experiments, setExperiments] = useState<any[]>([]);
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState('24h');
  const [refreshInterval, setRefreshInterval] = useState(30);
  const [selectedMetric, setSelectedMetric] = useState('all');
  const [selectedTab, setSelectedTab] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // 加载实验列表
  const loadExperiments = useCallback(async () => {
    try {
      const response = await experimentService.listExperiments({ status: 'active' });
      setExperiments(response.experiments);
      if (response.experiments.length > 0 && !selectedExperiment) {
        setSelectedExperiment(response.experiments[0].id);
      }
    } catch (err) {
      console.error('加载实验列表失败:', err);
    }
  }, [selectedExperiment]);

  // 加载仪表板数据
  const loadDashboardData = useCallback(async () => {
    if (!selectedExperiment) return;
    
    setLoading(true);
    setError(null);
    try {
      // 并行加载所有数据
      const [experiment, metrics, events, alerts, health] = await Promise.all([
        experimentService.getExperiment(selectedExperiment),
        metricsService.getExperimentMetrics(selectedExperiment, { timeRange }),
        experimentService.getExperimentEvents(selectedExperiment, { limit: 50 }),
        metricsService.getAlerts(selectedExperiment),
        metricsService.getHealthStatus(selectedExperiment)
      ]);

      // 模拟实时统计数据
      const realTimeStats = {
        currentUsers: Math.floor(Math.random() * 10000),
        requestsPerSecond: Math.floor(Math.random() * 1000),
        errorRate: Math.random() * 5,
        avgLatency: Math.random() * 200
      };

      setDashboardData({
        experiment,
        metrics,
        events,
        alerts,
        health,
        realTimeStats
      });
    } catch (err) {
      setError('加载仪表板数据失败');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [selectedExperiment, timeRange]);

  // 初始加载
  useEffect(() => {
    loadExperiments();
  }, [loadExperiments]);

  // 加载数据
  useEffect(() => {
    if (selectedExperiment) {
      loadDashboardData();
    }
  }, [selectedExperiment, loadDashboardData]);

  // 自动刷新
  useEffect(() => {
    if (autoRefresh && refreshInterval > 0) {
      refreshTimerRef.current = setInterval(() => {
        loadDashboardData();
      }, refreshInterval * 1000);

      return () => {
        if (refreshTimerRef.current) {
          clearInterval(refreshTimerRef.current);
        }
      };
    }
  }, [autoRefresh, refreshInterval, loadDashboardData]);

  // 切换全屏
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  // 生成时间序列数据
  const generateTimeSeriesData = () => {
    const data = [];
    const now = new Date();
    const points = timeRange === '1h' ? 12 : timeRange === '6h' ? 36 : 48;
    
    for (let i = points; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 5 * 60 * 1000);
      data.push({
        time: format(time, 'HH:mm'),
        control: Math.random() * 100,
        variant: Math.random() * 100 + 5,
        conversion: Math.random() * 10 + 45,
        revenue: Math.random() * 10000
      });
    }
    return data;
  };

  // 生成变体对比数据
  const generateVariantData = () => {
    if (!dashboardData?.experiment) return [];
    
    return dashboardData.experiment.variants.map((variant: any) => ({
      name: variant.name,
      users: Math.floor(Math.random() * 5000),
      conversion: Math.random() * 10 + 45,
      revenue: Math.random() * 50000,
      improvement: variant.isControl ? 0 : (Math.random() - 0.3) * 20
    }));
  };

  // 生成漏斗数据
  const generateFunnelData = () => [
    { name: '访问页面', value: 10000, percentage: 100 },
    { name: '查看产品', value: 7500, percentage: 75 },
    { name: '加入购物车', value: 4500, percentage: 45 },
    { name: '开始结算', value: 3000, percentage: 30 },
    { name: '完成购买', value: 1500, percentage: 15 }
  ];

  if (!dashboardData && !loading) {
    return (
      <Container maxWidth="xl" sx={{ py: 3 }}>
        <Alert severity="info">
          请选择一个运行中的实验来查看监控仪表板
        </Alert>
      </Container>
    );
  }

  return (
    <Box ref={containerRef} sx={{ bgcolor: 'background.default', minHeight: '100vh' }}>
      <Container maxWidth={isFullscreen ? false : 'xl'} sx={{ py: 3 }}>
        {/* 顶部控制栏 */}
        <Paper sx={{ p: 2, mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>选择实验</InputLabel>
                <Select
                  value={selectedExperiment}
                  onChange={(e) => setSelectedExperiment(e.target.value)}
                  label="选择实验"
                >
                  {experiments.map(exp => (
                    <MenuItem key={exp.id} value={exp.id}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Chip 
                          label={exp.status} 
                          size="small" 
                          color="success"
                        />
                        <span>{exp.name}</span>
                      </Stack>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={2}>
              <ToggleButtonGroup
                value={timeRange}
                exclusive
                onChange={(_, value) => value && setTimeRange(value)}
                size="small"
                fullWidth
              >
                {TIME_RANGES.map(range => (
                  <ToggleButton key={range.value} value={range.value}>
                    {range.label}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
            </Grid>
            <Grid item xs={12} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>刷新间隔</InputLabel>
                <Select
                  value={refreshInterval}
                  onChange={(e) => setRefreshInterval(Number(e.target.value))}
                  label="刷新间隔"
                >
                  {REFRESH_INTERVALS.map(interval => (
                    <MenuItem key={interval.value} value={interval.value}>
                      {interval.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={5}>
              <Stack direction="row" spacing={1} justifyContent="flex-end">
                <Button
                  startIcon={<RefreshIcon />}
                  onClick={loadDashboardData}
                  variant="outlined"
                  size="small"
                >
                  刷新
                </Button>
                <Button
                  startIcon={<DownloadIcon />}
                  variant="outlined"
                  size="small"
                >
                  导出报告
                </Button>
                <IconButton onClick={toggleFullscreen} size="small">
                  {isFullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
                </IconButton>
                <IconButton size="small">
                  <SettingsIcon />
                </IconButton>
              </Stack>
            </Grid>
          </Grid>
        </Paper>

        {/* 健康状态警告 */}
        {dashboardData?.health.status !== 'healthy' && (
          <Alert 
            severity={dashboardData.health.status === 'warning' ? 'warning' : 'error'}
            sx={{ mb: 3 }}
            action={
              <Button color="inherit" size="small">
                查看详情
              </Button>
            }
          >
            <strong>实验健康问题：</strong>
            {dashboardData.health.issues.join(', ')}
          </Alert>
        )}

        {/* 实时统计卡片 */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography color="text.secondary" gutterBottom variant="body2">
                      当前在线用户
                    </Typography>
                    <Typography variant="h4">
                      {dashboardData?.realTimeStats.currentUsers.toLocaleString()}
                    </Typography>
                    <Typography variant="body2" color="success.main">
                      <TrendingUpIcon fontSize="small" sx={{ verticalAlign: 'middle' }} />
                      +12.5%
                    </Typography>
                  </Box>
                  <Avatar sx={{ bgcolor: 'primary.light' }}>
                    <GroupIcon />
                  </Avatar>
                </Stack>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography color="text.secondary" gutterBottom variant="body2">
                      请求速率
                    </Typography>
                    <Typography variant="h4">
                      {dashboardData?.realTimeStats.requestsPerSecond}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      请求/秒
                    </Typography>
                  </Box>
                  <Avatar sx={{ bgcolor: 'success.light' }}>
                    <SpeedIcon />
                  </Avatar>
                </Stack>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography color="text.secondary" gutterBottom variant="body2">
                      错误率
                    </Typography>
                    <Typography variant="h4" color={dashboardData?.realTimeStats.errorRate! > 2 ? 'error.main' : 'text.primary'}>
                      {dashboardData?.realTimeStats.errorRate.toFixed(2)}%
                    </Typography>
                    <Typography variant="body2" color="error.main">
                      <TrendingUpIcon fontSize="small" sx={{ verticalAlign: 'middle' }} />
                      +0.5%
                    </Typography>
                  </Box>
                  <Avatar sx={{ bgcolor: 'error.light' }}>
                    <BugIcon />
                  </Avatar>
                </Stack>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography color="text.secondary" gutterBottom variant="body2">
                      平均延迟
                    </Typography>
                    <Typography variant="h4">
                      {dashboardData?.realTimeStats.avgLatency.toFixed(0)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      毫秒
                    </Typography>
                  </Box>
                  <Avatar sx={{ bgcolor: 'info.light' }}>
                    <TimelineIcon />
                  </Avatar>
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* 主要图表区域 */}
        <Grid container spacing={3}>
          {/* 转化率趋势 */}
          <Grid item xs={12} lg={8}>
            <Paper sx={{ p: 2, height: 400 }}>
              <Typography variant="h6" gutterBottom>
                转化率趋势
              </Typography>
              <ResponsiveContainer width="100%" height="90%">
                <LineChart data={generateTimeSeriesData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <ChartTooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="control" 
                    stroke={COLORS.control} 
                    name="对照组"
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="variant" 
                    stroke={COLORS.primary} 
                    name="实验组"
                    strokeWidth={2}
                  />
                  <ReferenceLine y={50} stroke="red" strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* 变体对比 */}
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 2, height: 400 }}>
              <Typography variant="h6" gutterBottom>
                变体表现
              </Typography>
              <ResponsiveContainer width="100%" height="90%">
                <BarChart data={generateVariantData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <ChartTooltip />
                  <Bar dataKey="conversion" fill={COLORS.primary} name="转化率%" />
                  <Bar dataKey="improvement" fill={COLORS.success} name="提升%" />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* 收入趋势 */}
          <Grid item xs={12} lg={6}>
            <Paper sx={{ p: 2, height: 350 }}>
              <Typography variant="h6" gutterBottom>
                收入趋势
              </Typography>
              <ResponsiveContainer width="100%" height="90%">
                <AreaChart data={generateTimeSeriesData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <ChartTooltip />
                  <Area 
                    type="monotone" 
                    dataKey="revenue" 
                    stroke={COLORS.success} 
                    fill={COLORS.success}
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* 转化漏斗 */}
          <Grid item xs={12} lg={6}>
            <Paper sx={{ p: 2, height: 350 }}>
              <Typography variant="h6" gutterBottom>
                转化漏斗
              </Typography>
              <List>
                {generateFunnelData().map((step, index) => (
                  <ListItem key={step.name}>
                    <ListItemText
                      primary={step.name}
                      secondary={`${step.value.toLocaleString()} 用户`}
                    />
                    <Box sx={{ flexGrow: 1, mx: 2 }}>
                      <LinearProgress 
                        variant="determinate" 
                        value={step.percentage}
                        sx={{ height: 10, borderRadius: 5 }}
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {step.percentage}%
                    </Typography>
                  </ListItem>
                ))}
              </List>
            </Paper>
          </Grid>

          {/* 实时事件流 */}
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 2, height: 400 }}>
              <Typography variant="h6" gutterBottom>
                实时事件
              </Typography>
              <List sx={{ maxHeight: 350, overflow: 'auto' }}>
                {dashboardData?.events.slice(0, 10).map((event: any, index) => (
                  <ListItem key={index}>
                    <ListItemAvatar>
                      <Avatar sx={{ bgcolor: event.type === 'error' ? 'error.main' : 'primary.main', width: 32, height: 32 }}>
                        {event.type === 'error' ? <ErrorIcon fontSize="small" /> : <CheckCircleIcon fontSize="small" />}
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={event.message}
                      secondary={format(new Date(event.timestamp), 'HH:mm:ss', { locale: zhCN })}
                    />
                  </ListItem>
                ))}
              </List>
            </Paper>
          </Grid>

          {/* 告警列表 */}
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 2, height: 400 }}>
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                <Typography variant="h6">
                  活跃告警
                </Typography>
                <Badge badgeContent={dashboardData?.alerts.length} color="error">
                  <NotificationsIcon />
                </Badge>
              </Stack>
              <List sx={{ maxHeight: 350, overflow: 'auto' }}>
                {dashboardData?.alerts.map((alert: any, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      {alert.severity === 'critical' ? (
                        <ErrorIcon color="error" />
                      ) : alert.severity === 'warning' ? (
                        <WarningIcon color="warning" />
                      ) : (
                        <InfoIcon color="info" />
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={alert.title}
                      secondary={alert.description}
                    />
                  </ListItem>
                ))}
                {dashboardData?.alerts.length === 0 && (
                  <ListItem>
                    <ListItemText
                      primary="没有活跃告警"
                      secondary="系统运行正常"
                    />
                  </ListItem>
                )}
              </List>
            </Paper>
          </Grid>

          {/* 指标详情表格 */}
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 2, height: 400 }}>
              <Typography variant="h6" gutterBottom>
                关键指标
              </Typography>
              <List>
                {dashboardData?.metrics.slice(0, 6).map((metric: any, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={metric.name}
                      secondary={
                        <Stack direction="row" spacing={1} alignItems="center">
                          <Typography variant="body2">
                            {metric.value.toFixed(2)} {metric.unit}
                          </Typography>
                          {metric.change !== 0 && (
                            <Chip
                              label={`${metric.change > 0 ? '+' : ''}${metric.change.toFixed(1)}%`}
                              size="small"
                              color={metric.change > 0 ? 'success' : 'error'}
                            />
                          )}
                        </Stack>
                      }
                    />
                    {metric.significant && (
                      <Chip label="显著" size="small" color="primary" />
                    )}
                  </ListItem>
                ))}
              </List>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default ExperimentDashboardPage;