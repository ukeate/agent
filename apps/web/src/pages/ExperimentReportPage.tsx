/**
 * 实验报告展示页面
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Stack,
  Divider,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  AlertTitle,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Tooltip,
  LinearProgress,
  CircularProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  ToggleButton,
  ToggleButtonGroup,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tab,
  Tabs,
  Box as MuiBox
} from '@mui/material';
import {
  Download as DownloadIcon,
  Print as PrintIcon,
  Share as ShareIcon,
  Email as EmailIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon,
  Science as ScienceIcon,
  Assessment as AssessmentIcon,
  Timeline as TimelineIcon,
  Group as GroupIcon,
  Speed as SpeedIcon,
  BugReport as BugIcon,
  Security as SecurityIcon,
  Lightbulb as LightbulbIcon,
  Flag as FlagIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
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
  Area,
  AreaChart,
  ScatterChart,
  Scatter,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import { useParams } from 'react-router-dom';
import { experimentService } from '../services/experimentService';
import { reportService } from '../services/reportService';

// 图表颜色
const COLORS = {
  primary: '#3f51b5',
  success: '#4caf50',
  warning: '#ff9800',
  error: '#f44336',
  info: '#2196f3',
  control: '#9e9e9e',
  variants: ['#673ab7', '#009688', '#ff5722', '#795548']
};

// 报告部分
interface ReportSection {
  id: string;
  title: string;
  icon: React.ReactElement;
  content: React.ReactNode;
}

// 报告数据
interface ReportData {
  experiment: any;
  summary: {
    status: string;
    startDate: Date;
    endDate?: Date;
    duration: number;
    totalUsers: number;
    totalEvents: number;
    recommendation: string;
  };
  metrics: {
    primary: any[];
    secondary: any[];
    guardrail: any[];
  };
  statistics: {
    confidence: number;
    power: number;
    sampleSize: {
      required: number;
      achieved: number;
    };
    srm: {
      passed: boolean;
      pValue: number;
    };
  };
  variants: any[];
  segments: any[];
  timeline: any[];
  insights: string[];
  recommendations: string[];
}

const ExperimentReportPage: React.FC = () => {
  const { experimentId } = useParams<{ experimentId: string }>();
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState(0);
  const [expandedSections, setExpandedSections] = useState<string[]>(['summary', 'metrics']);
  const [exportFormat, setExportFormat] = useState<'pdf' | 'html' | 'json'>('pdf');
  const reportRef = useRef<HTMLDivElement>(null);

  // 加载报告数据
  const loadReportData = useCallback(async () => {
    if (!experimentId) return;
    
    setLoading(true);
    setError(null);
    try {
      const report = await reportService.generateReport(experimentId);
      setReportData(report);
    } catch (err) {
      setError('加载报告失败');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [experimentId]);

  useEffect(() => {
    loadReportData();
  }, [loadReportData]);

  // 导出报告
  const exportReport = useCallback(async () => {
    if (!experimentId) return;
    
    try {
      const blob = await reportService.exportReport(experimentId, exportFormat);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `experiment-report-${experimentId}.${exportFormat}`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('导出失败:', err);
    }
  }, [experimentId, exportFormat]);

  // 打印报告
  const printReport = useCallback(() => {
    window.print();
  }, []);

  // 分享报告
  const shareReport = useCallback(async () => {
    if (!experimentId) return;
    
    try {
      const shareUrl = await reportService.createShareLink(experimentId);
      navigator.clipboard.writeText(shareUrl);
      // 显示复制成功提示
    } catch (err) {
      console.error('分享失败:', err);
    }
  }, [experimentId]);

  // 切换部分展开
  const toggleSection = useCallback((sectionId: string) => {
    setExpandedSections(prev =>
      prev.includes(sectionId)
        ? prev.filter(id => id !== sectionId)
        : [...prev, sectionId]
    );
  }, []);

  // 获取结果图标
  const getResultIcon = (improvement: number, significant: boolean) => {
    if (!significant) return <TrendingFlatIcon color="action" />;
    if (improvement > 0) return <TrendingUpIcon color="success" />;
    return <TrendingDownIcon color="error" />;
  };

  // 获取结果颜色
  const getResultColor = (improvement: number, significant: boolean) => {
    if (!significant) return 'default';
    if (improvement > 0) return 'success';
    return 'error';
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (error || !reportData) {
    return (
      <Container maxWidth="lg" sx={{ py: 3 }}>
        <Alert severity="error">
          {error || '无法加载报告'}
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 3 }}>
      <div ref={reportRef}>
        {/* 报告头部 */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={8}>
              <Stack spacing={1}>
                <Typography variant="h4" component="h1">
                  实验报告
                </Typography>
                <Typography variant="h6" color="text.secondary">
                  {reportData.experiment.name}
                </Typography>
                <Stack direction="row" spacing={1}>
                  <Chip
                    label={reportData.summary.status}
                    color={reportData.summary.status === '已完成' ? 'success' : 'primary'}
                    size="small"
                  />
                  <Chip
                    label={reportData.experiment.type}
                    variant="outlined"
                    size="small"
                  />
                  <Typography variant="body2" color="text.secondary">
                    {format(reportData.summary.startDate, 'yyyy-MM-dd', { locale: zhCN })} 
                    {reportData.summary.endDate && ` - ${format(reportData.summary.endDate, 'yyyy-MM-dd', { locale: zhCN })}`}
                  </Typography>
                </Stack>
              </Stack>
            </Grid>
            <Grid item xs={12} md={4}>
              <Stack direction="row" spacing={1} justifyContent="flex-end">
                <FormControl size="small" sx={{ minWidth: 100 }}>
                  <Select
                    value={exportFormat}
                    onChange={(e) => setExportFormat(e.target.value as any)}
                  >
                    <MenuItem value="pdf">PDF</MenuItem>
                    <MenuItem value="html">HTML</MenuItem>
                    <MenuItem value="json">JSON</MenuItem>
                  </Select>
                </FormControl>
                <Button
                  startIcon={<DownloadIcon />}
                  onClick={exportReport}
                  variant="outlined"
                >
                  导出
                </Button>
                <IconButton onClick={printReport}>
                  <PrintIcon />
                </IconButton>
                <IconButton onClick={shareReport}>
                  <ShareIcon />
                </IconButton>
              </Stack>
            </Grid>
          </Grid>
        </Paper>

        {/* 执行摘要 */}
        <Accordion
          expanded={expandedSections.includes('summary')}
          onChange={() => toggleSection('summary')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">执行摘要</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Alert 
                  severity={reportData.summary.recommendation === '建议采用' ? 'success' : 'info'}
                  icon={<LightbulbIcon />}
                >
                  <AlertTitle>建议</AlertTitle>
                  {reportData.summary.recommendation}
                </Alert>
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    关键发现
                  </Typography>
                  <List>
                    {reportData.insights.slice(0, 5).map((insight, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <CheckCircleIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText primary={insight} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              </Grid>
              <Grid item xs={12} md={4}>
                <Stack spacing={2}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        实验时长
                      </Typography>
                      <Typography variant="h5">
                        {reportData.summary.duration} 天
                      </Typography>
                    </CardContent>
                  </Card>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        总用户数
                      </Typography>
                      <Typography variant="h5">
                        {reportData.summary.totalUsers.toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        总事件数
                      </Typography>
                      <Typography variant="h5">
                        {reportData.summary.totalEvents.toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                </Stack>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* 指标结果 */}
        <Accordion
          expanded={expandedSections.includes('metrics')}
          onChange={() => toggleSection('metrics')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">指标结果</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Tabs value={selectedTab} onChange={(_, v) => setSelectedTab(v)}>
              <Tab label="主要指标" />
              <Tab label="次要指标" />
              <Tab label="护栏指标" />
            </Tabs>
            <Box sx={{ mt: 3 }}>
              {selectedTab === 0 && (
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>指标名称</TableCell>
                        <TableCell>对照组</TableCell>
                        <TableCell>实验组</TableCell>
                        <TableCell>提升</TableCell>
                        <TableCell>P值</TableCell>
                        <TableCell>置信区间</TableCell>
                        <TableCell>结果</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {reportData.metrics.primary.map((metric, index) => (
                        <TableRow key={index}>
                          <TableCell>{metric.name}</TableCell>
                          <TableCell>{metric.control.toFixed(2)}</TableCell>
                          <TableCell>{metric.treatment.toFixed(2)}</TableCell>
                          <TableCell>
                            <Stack direction="row" spacing={1} alignItems="center">
                              {getResultIcon(metric.improvement, metric.significant)}
                              <Typography
                                color={metric.improvement > 0 ? 'success.main' : 'error.main'}
                              >
                                {metric.improvement > 0 ? '+' : ''}{metric.improvement.toFixed(2)}%
                              </Typography>
                            </Stack>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={metric.pValue.toFixed(4)}
                              size="small"
                              color={metric.pValue < 0.05 ? 'success' : 'default'}
                            />
                          </TableCell>
                          <TableCell>
                            [{metric.ci_lower.toFixed(2)}, {metric.ci_upper.toFixed(2)}]
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={metric.significant ? '显著' : '不显著'}
                              size="small"
                              color={getResultColor(metric.improvement, metric.significant) as any}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
              {selectedTab === 1 && (
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>指标名称</TableCell>
                        <TableCell>对照组</TableCell>
                        <TableCell>实验组</TableCell>
                        <TableCell>提升</TableCell>
                        <TableCell>P值</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {reportData.metrics.secondary.map((metric, index) => (
                        <TableRow key={index}>
                          <TableCell>{metric.name}</TableCell>
                          <TableCell>{metric.control.toFixed(2)}</TableCell>
                          <TableCell>{metric.treatment.toFixed(2)}</TableCell>
                          <TableCell>
                            {metric.improvement > 0 ? '+' : ''}{metric.improvement.toFixed(2)}%
                          </TableCell>
                          <TableCell>{metric.pValue.toFixed(4)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
              {selectedTab === 2 && (
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>指标名称</TableCell>
                        <TableCell>对照组</TableCell>
                        <TableCell>实验组</TableCell>
                        <TableCell>变化</TableCell>
                        <TableCell>状态</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {reportData.metrics.guardrail.map((metric, index) => (
                        <TableRow key={index}>
                          <TableCell>{metric.name}</TableCell>
                          <TableCell>{metric.control.toFixed(2)}</TableCell>
                          <TableCell>{metric.treatment.toFixed(2)}</TableCell>
                          <TableCell>
                            {metric.change > 0 ? '+' : ''}{metric.change.toFixed(2)}%
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={metric.violated ? '违反' : '正常'}
                              size="small"
                              color={metric.violated ? 'error' : 'success'}
                              icon={metric.violated ? <WarningIcon /> : <CheckCircleIcon />}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* 统计分析 */}
        <Accordion
          expanded={expandedSections.includes('statistics')}
          onChange={() => toggleSection('statistics')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">统计分析</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      样本量分析
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        达成率
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={(reportData.statistics.sampleSize.achieved / reportData.statistics.sampleSize.required) * 100}
                        sx={{ height: 10, borderRadius: 5, mt: 1 }}
                      />
                    </Box>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          需要样本量
                        </Typography>
                        <Typography variant="h6">
                          {reportData.statistics.sampleSize.required.toLocaleString()}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          实际样本量
                        </Typography>
                        <Typography variant="h6">
                          {reportData.statistics.sampleSize.achieved.toLocaleString()}
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      统计检验
                    </Typography>
                    <List>
                      <ListItem>
                        <ListItemText
                          primary="置信水平"
                          secondary={`${reportData.statistics.confidence}%`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="统计功效"
                          secondary={`${reportData.statistics.power}%`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          {reportData.statistics.srm.passed ? (
                            <CheckCircleIcon color="success" />
                          ) : (
                            <CancelIcon color="error" />
                          )}
                        </ListItemIcon>
                        <ListItemText
                          primary="SRM检验"
                          secondary={`P值: ${reportData.statistics.srm.pValue.toFixed(4)}`}
                        />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* 数据可视化 */}
        <Accordion
          expanded={expandedSections.includes('visualization')}
          onChange={() => toggleSection('visualization')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">数据可视化</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              {/* 指标趋势图 */}
              <Grid item xs={12} lg={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    主要指标趋势
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={reportData.timeline}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <ChartTooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="control"
                        stroke={COLORS.control}
                        name="对照组"
                      />
                      <Line
                        type="monotone"
                        dataKey="treatment"
                        stroke={COLORS.primary}
                        name="实验组"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>

              {/* 变体对比图 */}
              <Grid item xs={12} lg={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    变体表现对比
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={reportData.variants}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <ChartTooltip />
                      <Legend />
                      <Bar dataKey="conversion" fill={COLORS.primary} name="转化率" />
                      <Bar dataKey="revenue" fill={COLORS.success} name="收入" />
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>

              {/* 用户分布饼图 */}
              <Grid item xs={12} lg={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    用户分布
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={reportData.variants}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={(entry) => `${entry.name}: ${entry.users}`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="users"
                      >
                        {reportData.variants.map((entry: any, index: number) => (
                          <Cell key={`cell-${index}`} fill={COLORS.variants[index % COLORS.variants.length]} />
                        ))}
                      </Pie>
                      <ChartTooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>

              {/* 分段分析雷达图 */}
              <Grid item xs={12} lg={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    分段表现
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={reportData.segments}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="segment" />
                      <PolarRadiusAxis />
                      <Radar
                        name="对照组"
                        dataKey="control"
                        stroke={COLORS.control}
                        fill={COLORS.control}
                        fillOpacity={0.6}
                      />
                      <Radar
                        name="实验组"
                        dataKey="treatment"
                        stroke={COLORS.primary}
                        fill={COLORS.primary}
                        fillOpacity={0.6}
                      />
                      <Legend />
                    </RadarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* 建议和后续步骤 */}
        <Accordion
          expanded={expandedSections.includes('recommendations')}
          onChange={() => toggleSection('recommendations')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">建议和后续步骤</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  <LightbulbIcon sx={{ verticalAlign: 'middle', mr: 1 }} />
                  建议
                </Typography>
                <List>
                  {reportData.recommendations.map((rec, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <FlagIcon color="primary" />
                      </ListItemIcon>
                      <ListItemText primary={rec} />
                    </ListItem>
                  ))}
                </List>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  <SecurityIcon sx={{ verticalAlign: 'middle', mr: 1 }} />
                  风险和注意事项
                </Typography>
                <Alert severity="warning">
                  <AlertTitle>潜在风险</AlertTitle>
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    <li>长期效应可能与短期结果不同</li>
                    <li>季节性因素可能影响结果</li>
                    <li>需要持续监控关键指标</li>
                  </ul>
                </Alert>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>
      </div>
    </Container>
  );
};

export default ExperimentReportPage;