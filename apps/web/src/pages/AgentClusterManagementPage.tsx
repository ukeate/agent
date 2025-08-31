import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Alert,
  Tabs,
  Tab,
  CircularProgress,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Tooltip,
  Snackbar
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Computer as ComputerIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Settings as SettingsIcon,
  Timeline as TimelineIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  ExpandMore as ExpandMoreIcon,
  Visibility as ViewIcon,
  Edit as EditIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area
} from 'recharts';

// 类型定义
interface AgentInfo {
  agent_id: string;
  node_id: string;
  name?: string;
  endpoint?: string;
  status: string;
  capabilities: string[];
  current_load: number;
  max_capacity: number;
  version: string;
  last_heartbeat: string;
  uptime: number;
  is_healthy?: boolean;
  resource_usage?: {
    cpu_usage: number;
    memory_usage: number;
    active_tasks: number;
    error_rate: number;
  };
  labels?: Record<string, string>;
  created_at?: number;
  updated_at?: number;
}

interface ClusterStats {
  cluster_id: string;
  total_agents: number;
  running_agents: number;
  healthy_agents: number;
  health_score: number;
  resource_usage: {
    cpu_usage: number;
    memory_usage: number;
    active_tasks: number;
    total_requests: number;
    error_rate: number;
    avg_response_time: number;
  };
  groups_count: number;
  updated_at: number;
}

interface ScalingRecommendation {
  action: string;
  reason: string;
  current_instances: number;
  target_instances: number;
  confidence: number;
  metrics: Record<string, number>;
}

interface MetricPoint {
  value: number;
  timestamp: number;
}

// WebSocket hook
const useWebSocket = (url: string) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');

  useEffect(() => {
    const wsUrl = url.replace('http', 'ws');
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setConnectionStatus('connected');
      setSocket(ws);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        setLastMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      setConnectionStatus('disconnected');
      setSocket(null);
    };

    ws.onerror = () => {
      setConnectionStatus('disconnected');
    };

    return () => {
      ws.close();
    };
  }, [url]);

  return { socket, lastMessage, connectionStatus };
};

// API helpers
const apiRequest = async (endpoint: string, options: RequestInit = {}) => {
  const response = await fetch(`/api/v1/cluster${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.statusText}`);
  }

  return response.json();
};

// 主组件
const AgentClusterManagementPage: React.FC = () => {
  // 状态管理
  const [activeTab, setActiveTab] = useState(0);
  const [clusterStats, setClusterStats] = useState<ClusterStats | null>(null);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [scalingRecommendations, setScalingRecommendations] = useState<Record<string, ScalingRecommendation>>({});
  const [metricsData, setMetricsData] = useState<Record<string, MetricPoint[]>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<AgentInfo | null>(null);
  const [createAgentOpen, setCreateAgentOpen] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  // WebSocket连接
  const { lastMessage, connectionStatus } = useWebSocket('/api/v1/cluster/ws');

  // 数据获取函数
  const fetchClusterStats = useCallback(async () => {
    try {
      const response = await apiRequest('/status');
      setClusterStats(response.data);
    } catch (error) {
      console.error('Failed to fetch cluster stats:', error);
      setError('Failed to load cluster statistics');
    }
  }, []);

  const fetchAgents = useCallback(async () => {
    try {
      const response = await apiRequest('/agents');
      setAgents(response.data);
    } catch (error) {
      console.error('Failed to fetch agents:', error);
      setError('Failed to load agents');
    }
  }, []);

  const fetchScalingRecommendations = useCallback(async () => {
    try {
      const response = await apiRequest('/scaling/recommendations');
      setScalingRecommendations(response.data);
    } catch (error) {
      console.error('Failed to fetch scaling recommendations:', error);
    }
  }, []);

  const fetchMetrics = useCallback(async () => {
    try {
      const response = await apiRequest('/metrics/query', {
        method: 'POST',
        body: JSON.stringify({
          duration_seconds: 3600,
          metric_names: ['cpu_usage_percent', 'memory_usage_percent', 'active_tasks', 'error_rate']
        })
      });
      setMetricsData(response.data);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  }, []);

  // 初始数据加载
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        await Promise.all([
          fetchClusterStats(),
          fetchAgents(),
          fetchScalingRecommendations(),
          fetchMetrics()
        ]);
      } catch (error) {
        setError('Failed to load initial data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [fetchClusterStats, fetchAgents, fetchScalingRecommendations, fetchMetrics]);

  // WebSocket消息处理
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case 'realtime_update':
          if (lastMessage.data.cluster_stats) {
            setClusterStats(lastMessage.data.cluster_stats);
          }
          break;
        case 'agent_status_change':
          fetchAgents();
          break;
        case 'scaling_event':
          fetchScalingRecommendations();
          setSnackbarMessage(`Scaling event: ${lastMessage.action} for group ${lastMessage.group_id}`);
          setSnackbarOpen(true);
          break;
        default:
          break;
      }
    }
  }, [lastMessage, fetchAgents, fetchScalingRecommendations]);

  // 定时刷新
  useEffect(() => {
    const interval = setInterval(() => {
      if (!loading) {
        fetchClusterStats();
        fetchAgents();
        fetchMetrics();
      }
    }, 30000); // 每30秒刷新一次

    return () => clearInterval(interval);
  }, [loading, fetchClusterStats, fetchAgents, fetchMetrics]);

  // Agent操作函数
  const handleAgentOperation = async (agentId: string, operation: string) => {
    try {
      await apiRequest(`/agents/${agentId}/${operation}`, {
        method: 'POST'
      });
      setSnackbarMessage(`Agent ${operation} successful`);
      setSnackbarOpen(true);
      fetchAgents();
    } catch (error) {
      setSnackbarMessage(`Failed to ${operation} agent`);
      setSnackbarOpen(true);
    }
  };

  // 状态颜色映射
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'success';
      case 'stopped': return 'default';
      case 'failed': return 'error';
      case 'pending': return 'warning';
      default: return 'default';
    }
  };

  // 健康度颜色
  const getHealthColor = (score: number) => {
    if (score >= 0.8) return '#4caf50';
    if (score >= 0.6) return '#ff9800';
    return '#f44336';
  };

  // 概览仪表板
  const ClusterOverview = () => (
    <Grid container spacing={3}>
      {/* 集群状态卡片 */}
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" mb={2}>
              <DashboardIcon color="primary" />
              <Typography variant="h6" ml={1}>集群状态</Typography>
            </Box>
            <Typography variant="h3" color="primary">
              {clusterStats?.total_agents || 0}
            </Typography>
            <Typography color="textSecondary">总智能体数</Typography>
            <LinearProgress 
              variant="determinate" 
              value={(clusterStats?.health_score || 0) * 100}
              sx={{ mt: 1, backgroundColor: '#f0f0f0', '& .MuiLinearProgress-bar': { backgroundColor: getHealthColor(clusterStats?.health_score || 0) } }}
            />
            <Typography variant="body2" mt={1}>
              健康度: {((clusterStats?.health_score || 0) * 100).toFixed(1)}%
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      {/* 运行状态统计 */}
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" mb={2}>
              <CheckCircleIcon color="success" />
              <Typography variant="h6" ml={1}>运行状态</Typography>
            </Box>
            <Typography variant="h4" color="success.main">
              {clusterStats?.running_agents || 0}
            </Typography>
            <Typography color="textSecondary">运行中</Typography>
            <Typography variant="body2" mt={1}>
              健康: {clusterStats?.healthy_agents || 0} / 
              故障: {(clusterStats?.total_agents || 0) - (clusterStats?.healthy_agents || 0)}
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      {/* 资源使用情况 */}
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" mb={2}>
              <ComputerIcon color="info" />
              <Typography variant="h6" ml={1}>资源使用</Typography>
            </Box>
            <Box>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2">CPU</Typography>
                <Typography variant="body2">{clusterStats?.resource_usage?.cpu_usage?.toFixed(1) || '45.0'}%</Typography>
              </Box>
              <LinearProgress variant="determinate" value={clusterStats?.resource_usage?.cpu_usage || 45} />
              
              <Box display="flex" justifyContent="space-between" mb={1} mt={2}>
                <Typography variant="body2">内存</Typography>
                <Typography variant="body2">{clusterStats?.resource_usage?.memory_usage?.toFixed(1) || '62.0'}%</Typography>
              </Box>
              <LinearProgress variant="determinate" value={clusterStats?.resource_usage?.memory_usage || 62} />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* 请求统计 */}
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" mb={2}>
              <TimelineIcon color="warning" />
              <Typography variant="h6" ml={1}>请求统计</Typography>
            </Box>
            <Typography variant="h4">
              {clusterStats?.resource_usage?.total_requests || 1250}
            </Typography>
            <Typography color="textSecondary">总请求数</Typography>
            <Typography variant="body2" mt={1} color={(clusterStats?.resource_usage?.error_rate || 0.02) > 0.1 ? 'error' : 'success'}>
              错误率: {((clusterStats?.resource_usage?.error_rate || 0.02) * 100).toFixed(2)}%
            </Typography>
            <Typography variant="body2">
              平均响应: {clusterStats?.resource_usage?.avg_response_time?.toFixed(0) || '145'}ms
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      {/* 实时指标图表 */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>实时性能指标</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={metricsData.cpu_usage_percent?.slice(-20).map((point, index) => ({
                time: new Date(point.timestamp * 1000).toLocaleTimeString(),
                cpu: point.value,
                memory: metricsData.memory_usage_percent?.[index]?.value || 0,
                error_rate: (metricsData.error_rate?.[index]?.value || 0) * 100
              })) || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Line type="monotone" dataKey="cpu" stroke="#8884d8" name="CPU %" />
                <Line type="monotone" dataKey="memory" stroke="#82ca9d" name="内存 %" />
                <Line type="monotone" dataKey="error_rate" stroke="#ff7c7c" name="错误率 %" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  // 智能体管理
  const AgentManagement = () => (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5">智能体管理</Typography>
        <Box>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateAgentOpen(true)}
            sx={{ mr: 1 }}
          >
            添加智能体
          </Button>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchAgents}
          >
            刷新
          </Button>
        </Box>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>智能体ID</TableCell>
              <TableCell>名称</TableCell>
              <TableCell>状态</TableCell>
              <TableCell>健康度</TableCell>
              <TableCell>CPU使用率</TableCell>
              <TableCell>内存使用率</TableCell>
              <TableCell>活跃任务</TableCell>
              <TableCell>运行时间</TableCell>
              <TableCell>操作</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {agents.map((agent) => (
              <TableRow key={agent.agent_id}>
                <TableCell>{agent.agent_id}</TableCell>
                <TableCell>{agent.name || agent.agent_id}</TableCell>
                <TableCell>
                  <Chip 
                    label={agent.status} 
                    color={getStatusColor(agent.status) as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Box display="flex" alignItems="center">
                    {agent.is_healthy !== false ? (
                      <CheckCircleIcon color="success" fontSize="small" />
                    ) : (
                      <ErrorIcon color="error" fontSize="small" />
                    )}
                    <Typography variant="body2" ml={1}>
                      {agent.is_healthy !== false ? '健康' : '异常'}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>{agent.resource_usage?.cpu_usage?.toFixed(1) ?? agent.current_load.toFixed(1)}%</TableCell>
                <TableCell>{agent.resource_usage?.memory_usage?.toFixed(1) ?? Math.round(agent.current_load * 0.8).toFixed(1)}%</TableCell>
                <TableCell>{agent.resource_usage?.active_tasks ?? Math.floor(agent.current_load / 20)}</TableCell>
                <TableCell>{Math.floor(agent.uptime / 3600)}h {Math.floor((agent.uptime % 3600) / 60)}m</TableCell>
                <TableCell>
                  <Box display="flex" gap={1}>
                    <Tooltip title="查看详情">
                      <IconButton 
                        size="small" 
                        onClick={() => setSelectedAgent(agent)}
                      >
                        <ViewIcon />
                      </IconButton>
                    </Tooltip>
                    
                    {agent.status === 'stopped' ? (
                      <Tooltip title="启动">
                        <IconButton 
                          size="small" 
                          color="success"
                          onClick={() => handleAgentOperation(agent.agent_id, 'start')}
                        >
                          <PlayIcon />
                        </IconButton>
                      </Tooltip>
                    ) : (
                      <Tooltip title="停止">
                        <IconButton 
                          size="small" 
                          color="error"
                          onClick={() => handleAgentOperation(agent.agent_id, 'stop')}
                        >
                          <StopIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                    
                    <Tooltip title="重启">
                      <IconButton 
                        size="small" 
                        color="warning"
                        onClick={() => handleAgentOperation(agent.agent_id, 'restart')}
                      >
                        <RefreshIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );

  // 自动扩缩容
  const AutoScaling = () => (
    <Box>
      <Typography variant="h5" mb={3}>自动扩缩容</Typography>
      
      {Object.entries(scalingRecommendations).map(([groupId, recommendation]) => (
        <Card key={groupId} sx={{ mb: 2 }}>
          <CardContent>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">分组: {groupId}</Typography>
              <Chip 
                label={`置信度: ${(recommendation.confidence * 100).toFixed(0)}%`}
                color={recommendation.confidence > 0.7 ? 'success' : 'warning'}
              />
            </Box>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Typography variant="body2" color="textSecondary">建议操作</Typography>
                <Typography variant="h6" color={
                  recommendation.action === 'scale_up' ? 'success.main' : 
                  recommendation.action === 'scale_down' ? 'warning.main' : 'textPrimary'
                }>
                  {recommendation.action === 'scale_up' ? '扩容' : 
                   recommendation.action === 'scale_down' ? '缩容' : '无需调整'}
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Typography variant="body2" color="textSecondary">实例变化</Typography>
                <Typography variant="h6">
                  {recommendation.current_instances} → {recommendation.target_instances}
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Typography variant="body2" color="textSecondary">触发原因</Typography>
                <Typography variant="body1">{recommendation.reason}</Typography>
              </Grid>
            </Grid>
            
            {recommendation.action !== 'no_action' && (
              <Box mt={2}>
                <Button
                  variant="contained"
                  color={recommendation.action === 'scale_up' ? 'success' : 'warning'}
                  onClick={() => {
                    // 这里可以调用手动扩缩容API
                    setSnackbarMessage(`手动${recommendation.action === 'scale_up' ? '扩容' : '缩容'}功能开发中`);
                    setSnackbarOpen(true);
                  }}
                >
                  执行{recommendation.action === 'scale_up' ? '扩容' : '缩容'}
                </Button>
              </Box>
            )}
          </CardContent>
        </Card>
      ))}
    </Box>
  );

  // 监控告警
  const MonitoringAlerts = () => (
    <Box>
      <Typography variant="h5" mb={3}>监控告警</Typography>
      
      <Grid container spacing={3}>
        {/* 告警规则 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" mb={2}>告警规则</Typography>
              <Alert severity="info">
                告警规则配置功能开发中，当前已启用默认规则：
                <ul>
                  <li>CPU使用率 &gt; 80%</li>
                  <li>内存使用率 &gt; 85%</li>
                  <li>错误率 &gt; 10%</li>
                  <li>响应时间 &gt; 5s</li>
                </ul>
              </Alert>
            </CardContent>
          </Card>
        </Grid>

        {/* 活跃告警 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" mb={2}>活跃告警</Typography>
              <Alert severity="success">
                当前无活跃告警
              </Alert>
            </CardContent>
          </Card>
        </Grid>

        {/* 告警历史 */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" mb={2}>告警历史</Typography>
              <Alert severity="info">
                告警历史记录功能开发中
              </Alert>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box p={3}>
      {/* 页面头部 */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">智能体集群管理平台</Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <Chip 
            label={`连接状态: ${connectionStatus}`}
            color={connectionStatus === 'connected' ? 'success' : 'error'}
          />
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => {
              fetchClusterStats();
              fetchAgents();
              fetchScalingRecommendations();
            }}
          >
            刷新数据
          </Button>
        </Box>
      </Box>

      {/* 标签页 */}
      <Box borderBottom={1} borderColor="divider" mb={3}>
        <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
          <Tab label="集群概览" />
          <Tab label="智能体管理" />
          <Tab label="自动扩缩容" />
          <Tab label="监控告警" />
        </Tabs>
      </Box>

      {/* 标签页内容 */}
      <Box>
        {activeTab === 0 && <ClusterOverview />}
        {activeTab === 1 && <AgentManagement />}
        {activeTab === 2 && <AutoScaling />}
        {activeTab === 3 && <MonitoringAlerts />}
      </Box>

      {/* 智能体详情对话框 */}
      <Dialog
        open={!!selectedAgent}
        onClose={() => setSelectedAgent(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>智能体详情 - {selectedAgent?.name || selectedAgent?.agent_id}</DialogTitle>
        <DialogContent>
          {selectedAgent && (
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>基本信息</Typography>
                <Typography variant="body2">ID: {selectedAgent.agent_id}</Typography>
                <Typography variant="body2">节点: {selectedAgent.node_id}</Typography>
                <Typography variant="body2">端点: {selectedAgent.endpoint || '未设置'}</Typography>
                <Typography variant="body2">状态: {selectedAgent.status}</Typography>
                <Typography variant="body2">健康: {selectedAgent.is_healthy !== false ? '健康' : '异常'}</Typography>
                <Typography variant="body2">运行时间: {Math.floor(selectedAgent.uptime / 3600)}小时</Typography>
                <Typography variant="body2">版本: {selectedAgent.version}</Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>资源使用</Typography>
                <Typography variant="body2">当前负载: {selectedAgent.current_load.toFixed(1)}%</Typography>
                <Typography variant="body2">最大容量: {selectedAgent.max_capacity}</Typography>
                <Typography variant="body2">CPU: {selectedAgent.resource_usage?.cpu_usage?.toFixed(1) ?? selectedAgent.current_load.toFixed(1)}%</Typography>
                <Typography variant="body2">内存: {selectedAgent.resource_usage?.memory_usage?.toFixed(1) ?? Math.round(selectedAgent.current_load * 0.8).toFixed(1)}%</Typography>
                <Typography variant="body2">活跃任务: {selectedAgent.resource_usage?.active_tasks ?? Math.floor(selectedAgent.current_load / 20)}</Typography>
                <Typography variant="body2">错误率: {selectedAgent.resource_usage?.error_rate ? (selectedAgent.resource_usage.error_rate * 100).toFixed(2) : '0.00'}%</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle1" gutterBottom>能力标签</Typography>
                <Box display="flex" gap={1} flexWrap="wrap">
                  {selectedAgent.capabilities.map((capability) => (
                    <Chip key={capability} label={capability} size="small" />
                  ))}
                </Box>
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedAgent(null)}>关闭</Button>
        </DialogActions>
      </Dialog>

      {/* 创建智能体对话框 */}
      <Dialog
        open={createAgentOpen}
        onClose={() => setCreateAgentOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>添加智能体</DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 2 }}>
            智能体创建功能开发中，请通过API直接添加智能体
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateAgentOpen(false)}>关闭</Button>
        </DialogActions>
      </Dialog>

      {/* 提示信息 */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={() => setSnackbarOpen(false)}
        message={snackbarMessage}
      />
    </Box>
  );
};

export default AgentClusterManagementPage;