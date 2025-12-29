import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Tab,
  Tabs,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  Alert,
  CircularProgress,
  LinearProgress,
  IconButton,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  FormControlLabel,
  Switch,
  Slider
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  Visibility as VisibilityIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Assessment as AssessmentIcon,
  Settings as SettingsIcon,
  Compare as CompareIcon,
  TrendingUp as TrendingUpIcon,
  Psychology as PsychologyIcon,
  Security as SecurityIcon,
  Group as GroupIcon,
  AutoMode as AutoModeIcon,
  Timeline as TimelineIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { trainingDataServiceEnhanced } from '../services/trainingDataServiceEnhanced';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index, ...other }: TabPanelProps) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`training-data-enhanced-tabpanel-${index}`}
      aria-labelledby={`training-data-enhanced-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export default function TrainingDataManagementPageEnhanced() {
  const location = useLocation();
  const [currentTab, setCurrentTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 质量分析相关状态
  const [qualityAnalysis, setQualityAnalysis] = useState<any>(null);
  const [qualityTrends, setQualityTrends] = useState<any>(null);
  const [batchQualityMetrics, setBatchQualityMetrics] = useState<any>(null);

  // 智能标注相关状态
  const [intelligentTasks, setIntelligentTasks] = useState<any[]>([]);
  const [annotationInsights, setAnnotationInsights] = useState<any>(null);
  const [workflowOptimization, setWorkflowOptimization] = useState<any>(null);

  // 预处理管道相关状态
  const [processingPipelines, setProcessingPipelines] = useState<any[]>([]);
  const [pipelineStatus, setPipelineStatus] = useState<any>(null);
  const [pipelineValidation, setPipelineValidation] = useState<any>(null);

  // 版本管理相关状态
  const [versioningConfigs, setVersioningConfigs] = useState<any[]>([]);
  const [versionComparison, setVersionComparison] = useState<any>(null);
  const [impactPrediction, setImpactPrediction] = useState<any>(null);

  // 治理和合规性相关状态
  const [governancePolicies, setGovernancePolicies] = useState<any[]>([]);
  const [complianceAudits, setComplianceAudits] = useState<any[]>([]);

  // 分析和洞察相关状态
  const [datasetInsights, setDatasetInsights] = useState<any>(null);
  const [crossDatasetAnalysis, setCrossDatasetAnalysis] = useState<any>(null);

  // 工作流和协作相关状态
  const [automatedWorkflows, setAutomatedWorkflows] = useState<any[]>([]);
  const [workflowPerformance, setWorkflowPerformance] = useState<any>(null);
  const [collaborationMetrics, setCollaborationMetrics] = useState<any>(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  // 高级质量分析
  const performAdvancedQualityAnalysis = async (datasetId: string) => {
    try {
      setLoading(true);
      const analysis = await trainingDataServiceEnhanced.performAdvancedQualityAnalysis(datasetId, {
        analysis_types: ['completeness', 'accuracy', 'consistency', 'semantic_quality'],
        thresholds: { minimum_quality: 0.8, consistency_threshold: 0.9 },
        sample_size: 1000,
        include_recommendations: true
      });
      setQualityAnalysis(analysis);
    } catch (err) {
      setError(err instanceof Error ? err.message : '质量分析失败');
    } finally {
      setLoading(false);
    }
  };

  const trackQualityTrends = async (datasetId: string) => {
    try {
      const trends = await trainingDataServiceEnhanced.trackQualityTrends(datasetId, {
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        granularity: 'week'
      });
      setQualityTrends(trends);
    } catch (err) {
      setError(err instanceof Error ? err.message : '趋势分析失败');
    }
  };

  // 智能标注任务
  const createIntelligentAnnotationTask = async () => {
    try {
      setLoading(true);
      const task = await trainingDataServiceEnhanced.createIntelligentAnnotationTask({
        name: '智能文本分类标注',
        dataset_id: 'dataset_001',
        annotation_type: 'classification',
        ai_assistance_level: 'pre_annotation',
        quality_requirements: {
          minimum_agreement: 0.8,
          confidence_threshold: 0.9,
          review_percentage: 0.1
        },
        workflow_settings: {
          annotation_rounds: 2,
          validation_strategy: 'cross_validation',
          conflict_resolution: 'expert_review'
        },
        guidelines: '详细的标注指南...',
        examples: []
      });
      setIntelligentTasks(prev => [...prev, task]);
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建智能标注任务失败');
    } finally {
      setLoading(false);
    }
  };

  // 预处理管道管理
  const createAdvancedPipeline = async () => {
    try {
      setLoading(true);
      const pipeline = await trainingDataServiceEnhanced.createAdvancedPreprocessingPipeline({
        name: '高级文本预处理管道',
        description: '包含清洗、标准化、质量检查的完整预处理管道',
        input_sources: ['source_001', 'source_002'],
        processing_steps: [
          {
            step_id: 'text_cleaning',
            processor_type: 'text_cleaner',
            parameters: { remove_html: true, normalize_whitespace: true },
            parallel_execution: true,
            error_handling: 'skip'
          },
          {
            step_id: 'quality_check',
            processor_type: 'quality_validator',
            parameters: { min_length: 10, max_length: 1000 },
            parallel_execution: false,
            error_handling: 'fail_pipeline'
          }
        ],
        output_config: {
          format: 'jsonl',
          validation_rules: [],
          quality_gates: []
        },
        scheduling: {
          trigger_type: 'manual'
        }
      });
      setProcessingPipelines(prev => [...prev, pipeline]);
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建预处理管道失败');
    } finally {
      setLoading(false);
    }
  };

  // 智能版本管理
  const setupIntelligentVersioning = async (datasetName: string) => {
    try {
      const config = await trainingDataServiceEnhanced.createIntelligentVersioning(datasetName, {
        versioning_type: 'semantic',
        auto_versioning_rules: {
          trigger_conditions: ['quality_improvement', 'size_threshold'],
          version_increment_logic: { major: 'breaking_changes', minor: 'feature_additions', patch: 'bug_fixes' }
        },
        metadata_tracking: {
          track_lineage: true,
          track_transformations: true,
          track_quality_metrics: true,
          custom_metadata: {}
        },
        retention_policy: {
          max_versions: 10,
          retention_period: '1year',
          archival_strategy: 'compress_old_versions'
        }
      });
      setVersioningConfigs(prev => [...prev, config]);
    } catch (err) {
      setError(err instanceof Error ? err.message : '设置智能版本管理失败');
    }
  };

  // 数据治理
  const establishDataGovernance = async (datasetId: string) => {
    try {
      const governance = await trainingDataServiceEnhanced.establishDataGovernance(datasetId, {
        data_classification: 'internal',
        privacy_requirements: {
          contains_pii: false,
          anonymization_level: 'none',
          retention_requirements: {}
        },
        access_controls: {
          authorized_roles: ['data_scientist', 'ml_engineer'],
          access_restrictions: [],
          audit_requirements: {}
        },
        compliance_frameworks: ['GDPR', 'CCPA']
      });
      setGovernancePolicies(prev => [...prev, governance]);
    } catch (err) {
      setError(err instanceof Error ? err.message : '建立数据治理失败');
    }
  };

  // 数据洞察分析
  const generateDatasetInsights = async (datasetId: string) => {
    try {
      const insights = await trainingDataServiceEnhanced.generateDatasetInsights(datasetId, {
        analysis_types: ['statistical', 'quality', 'usage_patterns'],
        statistical_depth: 'advanced',
        include_predictions: true,
        comparison_datasets: []
      });
      setDatasetInsights(insights);
    } catch (err) {
      setError(err instanceof Error ? err.message : '生成数据洞察失败');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
      case 'pass':
        return <CheckCircleIcon color="success" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'failed':
      case 'error':
        return <ErrorIcon color="error" />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        增强训练数据管理系统
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 3 }}>
        企业级训练数据管理平台，提供智能化、自动化的数据处理和质量管理功能
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs 
          value={currentTab} 
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab icon={<PsychologyIcon />} label="智能质量分析" />
          <Tab icon={<EditIcon />} label="智能标注系统" />
          <Tab icon={<SettingsIcon />} label="高级预处理管道" />
          <Tab icon={<CompareIcon />} label="智能版本管理" />
          <Tab icon={<SecurityIcon />} label="数据治理合规" />
          <Tab icon={<TrendingUpIcon />} label="深度分析洞察" />
          <Tab icon={<AutoModeIcon />} label="自动化工作流" />
          <Tab icon={<GroupIcon />} label="团队协作管理" />
        </Tabs>
      </Box>

      {/* 智能质量分析 */}
      <TabPanel value={currentTab} index={0}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            智能质量分析系统
          </Typography>
          <Typography variant="body2" color="text.secondary">
            基于AI的数据质量评估，提供深度分析、趋势预测和自动化质量改进建议
          </Typography>
        </Box>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  质量分析控制台
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <TextField
                    fullWidth
                    size="small"
                    label="数据集ID"
                    defaultValue="dataset_001"
                    sx={{ mb: 2 }}
                  />
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<PsychologyIcon />}
                    onClick={() => performAdvancedQualityAnalysis('dataset_001')}
                    disabled={loading}
                    sx={{ mb: 1 }}
                  >
                    启动深度质量分析
                  </Button>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<TrendingUpIcon />}
                    onClick={() => trackQualityTrends('dataset_001')}
                    disabled={loading}
                  >
                    查看质量趋势
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={8}>
            {qualityAnalysis && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    质量分析报告
                  </Typography>
                  
                  <Grid container spacing={2} sx={{ mb: 3 }}>
                    {Object.entries(qualityAnalysis.quality_metrics).map(([metric, value]) => (
                      <Grid item xs={12} sm={6} md={4} key={metric}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h5" color="primary">
                            {((value as number) * 100).toFixed(1)}%
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {metric.replace('_', ' ').toUpperCase()}
                          </Typography>
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>

                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">质量问题详情</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      {qualityAnalysis.detailed_issues.map((issue: any, index: number) => (
                        <Alert
                          key={index}
                          severity={issue.severity === 'critical' ? 'error' : issue.severity as any}
                          sx={{ mb: 1 }}
                        >
                          <Typography variant="body2">
                            <strong>{issue.issue_type}:</strong> {issue.description}
                            (影响 {issue.count} 条记录)
                          </Typography>
                        </Alert>
                      ))}
                    </AccordionDetails>
                  </Accordion>

                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">改进建议</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List>
                        {qualityAnalysis.recommendations.quality_improvement_steps.map((step: string, index: number) => (
                          <ListItem key={index}>
                            <ListItemIcon>
                              <CheckCircleIcon color="success" />
                            </ListItemIcon>
                            <ListItemText primary={step} />
                          </ListItem>
                        ))}
                      </List>
                    </AccordionDetails>
                  </Accordion>
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>

        {qualityTrends && (
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                质量趋势分析
              </Typography>
              <Alert severity="info" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  总体趋势: <strong>{qualityTrends.trend_analysis.overall_trend}</strong>
                </Typography>
                <Typography variant="body2">
                  数据质量在过去一段时间内表现{qualityTrends.trend_analysis.overall_trend === 'improving' ? '改善' : 
                  qualityTrends.trend_analysis.overall_trend === 'declining' ? '下降' : '稳定'}
                </Typography>
              </Alert>
              
              <Box sx={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'grey.50', borderRadius: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  质量趋势图表 (集成图表库后显示实际数据可视化)
                </Typography>
              </Box>
            </CardContent>
          </Card>
        )}
      </TabPanel>

      {/* 智能标注系统 */}
      <TabPanel value={currentTab} index={1}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            智能标注系统
          </Typography>
          <Typography variant="body2" color="text.secondary">
            AI辅助的智能标注工作流，提供预标注、质量控制和协作标注功能
          </Typography>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  创建智能标注任务
                </Typography>
                
                <FormControl fullWidth margin="normal" size="small">
                  <InputLabel>标注类型</InputLabel>
                  <Select defaultValue="classification" label="标注类型">
                    <MenuItem value="classification">文本分类</MenuItem>
                    <MenuItem value="ner">命名实体识别</MenuItem>
                    <MenuItem value="relation_extraction">关系抽取</MenuItem>
                    <MenuItem value="summarization">摘要生成</MenuItem>
                  </Select>
                </FormControl>

                <FormControl fullWidth margin="normal" size="small">
                  <InputLabel>AI辅助级别</InputLabel>
                  <Select defaultValue="pre_annotation" label="AI辅助级别">
                    <MenuItem value="none">无AI辅助</MenuItem>
                    <MenuItem value="suggestions">AI建议</MenuItem>
                    <MenuItem value="pre_annotation">预标注</MenuItem>
                    <MenuItem value="active_learning">主动学习</MenuItem>
                  </Select>
                </FormControl>

                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" gutterBottom>质量要求</Typography>
                  <Typography variant="caption" color="text.secondary">
                    最小一致性: 80%
                  </Typography>
                  <Slider
                    defaultValue={80}
                    min={50}
                    max={100}
                    step={5}
                    marks
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${value}%`}
                  />
                </Box>

                <Button
                  fullWidth
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={createIntelligentAnnotationTask}
                  disabled={loading}
                  sx={{ mt: 2 }}
                >
                  创建智能标注任务
                </Button>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  智能标注任务列表
                </Typography>
                
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>任务名称</TableCell>
                        <TableCell>类型</TableCell>
                        <TableCell>AI辅助</TableCell>
                        <TableCell>预计完成时间</TableCell>
                        <TableCell>成本估算</TableCell>
                        <TableCell>状态</TableCell>
                        <TableCell>操作</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {intelligentTasks.map((task, index) => (
                        <TableRow key={index}>
                          <TableCell>{task.name || '智能文本分类标注'}</TableCell>
                          <TableCell>
                            <Chip label="分类标注" size="small" color="primary" />
                          </TableCell>
                          <TableCell>
                            <Chip label="预标注" size="small" color="success" />
                          </TableCell>
                          <TableCell>{task.estimated_completion_time || 120} 分钟</TableCell>
                          <TableCell>¥{task.cost_estimate || 150}</TableCell>
                          <TableCell>
                            <Chip label="进行中" size="small" color="warning" />
                          </TableCell>
                          <TableCell>
                            <IconButton size="small" color="primary">
                              <VisibilityIcon />
                            </IconButton>
                            <IconButton size="small" color="success">
                              <TrendingUpIcon />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                      {intelligentTasks.length === 0 && (
                        <TableRow>
                          <TableCell colSpan={7} align="center">
                            <Typography variant="body2" color="text.secondary">
                              暂无智能标注任务
                            </Typography>
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {annotationInsights && (
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                标注洞察报告
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>生产力指标</Typography>
                    <Typography variant="body2">
                      标注速度: {annotationInsights.productivity_metrics.annotations_per_hour}/小时
                    </Typography>
                    <Typography variant="body2">
                      质量一致性: {(annotationInsights.productivity_metrics.quality_consistency * 100).toFixed(1)}%
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>质量洞察</Typography>
                    <Typography variant="body2">
                      评审周转时间: {annotationInsights.quality_insights.review_turnaround_time}小时
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>改进建议</Typography>
                    <Typography variant="body2">
                      {annotationInsights.quality_insights.improvement_suggestions.length} 条建议
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        )}
      </TabPanel>

      {/* 高级预处理管道 */}
      <TabPanel value={currentTab} index={2}>
        <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h6" gutterBottom>
              高级预处理管道
            </Typography>
            <Typography variant="body2" color="text.secondary">
              可视化构建和管理复杂的数据预处理工作流
            </Typography>
          </Box>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={createAdvancedPipeline}
            disabled={loading}
          >
            创建管道
          </Button>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  管道配置
                </Typography>
                
                <Alert severity="info" sx={{ mb: 2 }}>
                  配置多步骤的数据预处理管道，支持并行执行和错误处理
                </Alert>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>处理步骤</Typography>
                  <List dense>
                    <ListItem>
                      <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                      <ListItemText 
                        primary="文本清洗" 
                        secondary="去除HTML标签、标准化空白字符"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                      <ListItemText 
                        primary="质量检查" 
                        secondary="长度验证、格式检查"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><InfoIcon color="info" /></ListItemIcon>
                      <ListItemText 
                        primary="语言检测" 
                        secondary="自动检测文本语言"
                      />
                    </ListItem>
                  </List>
                </Box>

                <Box>
                  <Typography variant="subtitle2" gutterBottom>调度配置</Typography>
                  <FormControlLabel
                    control={<Switch defaultChecked />}
                    label="启用自动调度"
                  />
                  <FormControlLabel
                    control={<Switch />}
                    label="并行执行"
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  管道执行状态
                </Typography>
                
                {processingPipelines.length > 0 ? (
                  <Box>
                    <LinearProgress variant="determinate" value={65} sx={{ mb: 2 }} />
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      当前进度: 65% (1,235/1,900 条记录)
                    </Typography>
                    
                    <Grid container spacing={2} sx={{ mt: 1 }}>
                      <Grid item xs={6}>
                        <Paper sx={{ p: 1, textAlign: 'center' }}>
                          <Typography variant="h6" color="success.main">120</Typography>
                          <Typography variant="caption">处理速度/分钟</Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={6}>
                        <Paper sx={{ p: 1, textAlign: 'center' }}>
                          <Typography variant="h6" color="info.main">98.5%</Typography>
                          <Typography variant="caption">成功率</Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                  </Box>
                ) : (
                  <Alert severity="info">
                    暂无运行中的预处理管道
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              管道历史记录
            </Typography>
            
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>管道名称</TableCell>
                    <TableCell>状态</TableCell>
                    <TableCell>处理记录数</TableCell>
                    <TableCell>执行时间</TableCell>
                    <TableCell>成功率</TableCell>
                    <TableCell>创建时间</TableCell>
                    <TableCell>操作</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>高级文本预处理管道</TableCell>
                    <TableCell>
                      <Chip label="运行中" size="small" color="warning" />
                    </TableCell>
                    <TableCell>1,235 / 1,900</TableCell>
                    <TableCell>45 分钟</TableCell>
                    <TableCell>98.5%</TableCell>
                    <TableCell>2024-01-15 14:30</TableCell>
                    <TableCell>
                      <IconButton size="small" color="primary">
                        <VisibilityIcon />
                      </IconButton>
                      <IconButton size="small" color="success">
                        <TimelineIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      </TabPanel>

      {/* 其他Tab面板的内容... */}
      {/* 为了简化展示，这里省略了其他Tab的详细实现 */}
      {[3, 4, 5, 6, 7].map((tabIndex) => (
        <TabPanel key={tabIndex} value={currentTab} index={tabIndex}>
          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              {tabIndex === 3 && '智能版本管理'}
              {tabIndex === 4 && '数据治理合规'}
              {tabIndex === 5 && '深度分析洞察'}
              {tabIndex === 6 && '自动化工作流'}
              {tabIndex === 7 && '团队协作管理'}
            </Typography>
            <Typography variant="body2">
              此模块展示了training_data模块的高级功能，包括智能化、自动化的数据管理能力。
              实际使用时这些功能会调用相应的API端点提供完整的企业级数据管理体验。
            </Typography>
          </Alert>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    功能特性
                  </Typography>
                  <List>
                    {tabIndex === 3 && (
                      <>
                        <ListItem><ListItemText primary="语义版本管理" secondary="自动版本号生成和冲突检测" /></ListItem>
                        <ListItem><ListItemText primary="影响预测分析" secondary="版本变更影响评估和回滚计划" /></ListItem>
                        <ListItem><ListItemText primary="智能比较分析" secondary="多维度版本差异分析" /></ListItem>
                      </>
                    )}
                    {tabIndex === 4 && (
                      <>
                        <ListItem><ListItemText primary="数据分类管理" secondary="自动数据敏感性分级" /></ListItem>
                        <ListItem><ListItemText primary="合规性审计" secondary="GDPR、CCPA等法规合规检查" /></ListItem>
                        <ListItem><ListItemText primary="访问控制" secondary="细粒度权限管理和审计追踪" /></ListItem>
                      </>
                    )}
                    {tabIndex === 5 && (
                      <>
                        <ListItem><ListItemText primary="深度统计分析" secondary="多维度数据特征分析" /></ListItem>
                        <ListItem><ListItemText primary="跨数据集分析" secondary="数据集间关联性和互补性分析" /></ListItem>
                        <ListItem><ListItemText primary="预测性洞察" secondary="数据趋势预测和改进建议" /></ListItem>
                      </>
                    )}
                    {tabIndex === 6 && (
                      <>
                        <ListItem><ListItemText primary="工作流自动化" secondary="基于事件触发的自动化处理" /></ListItem>
                        <ListItem><ListItemText primary="智能调度" secondary="资源优化和任务调度" /></ListItem>
                        <ListItem><ListItemText primary="异常处理" secondary="自动错误恢复和通知" /></ListItem>
                      </>
                    )}
                    {tabIndex === 7 && (
                      <>
                        <ListItem><ListItemText primary="团队协作" secondary="多人协作和权限管理" /></ListItem>
                        <ListItem><ListItemText primary="绩效分析" secondary="个人和团队生产力分析" /></ListItem>
                        <ListItem><ListItemText primary="项目管理" secondary="里程碑跟踪和进度管理" /></ListItem>
                      </>
                    )}
                  </List>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    快速操作
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Button variant="contained" startIcon={<SettingsIcon />}>
                      {tabIndex === 3 && '配置版本管理策略'}
                      {tabIndex === 4 && '设置治理策略'}
                      {tabIndex === 5 && '生成分析报告'}
                      {tabIndex === 6 && '创建自动化工作流'}
                      {tabIndex === 7 && '建立协作项目'}
                    </Button>
                    <Button variant="outlined" startIcon={<AssessmentIcon />}>
                      查看详细报告
                    </Button>
                    <Button variant="outlined" startIcon={<TrendingUpIcon />}>
                      查看历史趋势
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      ))}

      {loading && (
        <Box sx={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 9999 }}>
          <LinearProgress />
        </Box>
      )}
    </Box>
  );
}