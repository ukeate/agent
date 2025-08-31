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
  Tooltip
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
  Settings as SettingOutlined,
  DifferenceTwoTone as DiffOutlined
} from '@mui/icons-material';

interface DataSource {
  id: string;
  source_id: string;
  source_type: string;
  name: string;
  description: string;
  config: any;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

interface DataRecord {
  id: string;
  record_id: string;
  source_id: string;
  raw_data: any;
  processed_data: any;
  metadata: any;
  quality_score: number | null;
  status: string;
  created_at: string;
  processed_at: string | null;
}

interface AnnotationTask {
  id: string;
  task_id: string;
  name: string;
  description: string;
  task_type: string;
  data_record_count: number;
  assignee_count: number;
  status: string;
  created_by: string;
  created_at: string;
  deadline: string | null;
}

interface DataVersion {
  version_id: string;
  version_number: string;
  description: string;
  created_by: string;
  parent_version: string | null;
  record_count: number;
  size_bytes: number;
  data_hash: string;
  created_at: string;
  changes_summary: any;
  metadata: any;
}

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
      id={`training-data-tabpanel-${index}`}
      aria-labelledby={`training-data-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export default function TrainingDataManagementPage() {
  const location = useLocation();
  
  // 根据路由设置初始Tab
  const getInitialTab = () => {
    switch (location.pathname) {
      case '/training-data-management': return 0;
      case '/data-sources': return 1;
      case '/data-collection': return 2;
      case '/data-preprocessing': return 3;
      case '/data-annotation': return 4;
      case '/annotation-tasks': return 5;
      case '/annotation-quality': return 6;
      case '/data-versioning': return 7;
      case '/data-version-comparison': return 8;
      case '/data-export': return 9;
      case '/data-statistics': return 10;
      case '/quality-metrics': return 11;
      default: return 0;
    }
  };
  
  const [currentTab, setCurrentTab] = useState(getInitialTab());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // 数据源相关状态
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [dataSourceDialog, setDataSourceDialog] = useState(false);
  const [newDataSource, setNewDataSource] = useState({
    source_id: '',
    source_type: 'file',
    name: '',
    description: '',
    config: {}
  });
  
  // 数据记录相关状态
  const [dataRecords, setDataRecords] = useState<DataRecord[]>([]);
  const [recordsPage, setRecordsPage] = useState(0);
  const [recordsPerPage, setRecordsPerPage] = useState(10);
  const [recordsFilter, setRecordsFilter] = useState({
    source_id: '',
    status: '',
    min_quality_score: ''
  });
  
  // 标注任务相关状态
  const [annotationTasks, setAnnotationTasks] = useState<AnnotationTask[]>([]);
  const [taskDialog, setTaskDialog] = useState(false);
  const [newTask, setNewTask] = useState({
    name: '',
    description: '',
    task_type: 'text_classification',
    data_records: [] as string[],
    annotation_schema: {},
    guidelines: '',
    assignees: [] as string[],
    created_by: 'user_001'
  });
  
  // 版本管理相关状态
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [dataVersions, setDataVersions] = useState<DataVersion[]>([]);
  const [versionDialog, setVersionDialog] = useState(false);
  const [newVersion, setNewVersion] = useState({
    dataset_name: '',
    version_number: '',
    description: '',
    data_record_ids: [] as string[],
    parent_version: '',
    metadata: {}
  });
  
  // 统计信息
  const [statistics, setStatistics] = useState<any>(null);
  
  // 监听路由变化，更新当前Tab
  useEffect(() => {
    setCurrentTab(getInitialTab());
  }, [location.pathname]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  // API调用函数
  const fetchDataSources = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/training-data/sources');
      if (response.ok) {
        const data = await response.json();
        setDataSources(data);
      } else {
        throw new Error('获取数据源失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取数据源失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchDataRecords = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams({
        limit: recordsPerPage.toString(),
        offset: (recordsPage * recordsPerPage).toString(),
        ...(recordsFilter.source_id && { source_id: recordsFilter.source_id }),
        ...(recordsFilter.status && { status: recordsFilter.status }),
        ...(recordsFilter.min_quality_score && { min_quality_score: recordsFilter.min_quality_score })
      });
      
      const response = await fetch(`/api/v1/training-data/records?${params}`);
      if (response.ok) {
        const data = await response.json();
        setDataRecords(data.records);
      } else {
        throw new Error('获取数据记录失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取数据记录失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchAnnotationTasks = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/training-data/annotation-tasks');
      if (response.ok) {
        const data = await response.json();
        setAnnotationTasks(data.tasks);
      } else {
        throw new Error('获取标注任务失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取标注任务失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/training-data/datasets');
      if (response.ok) {
        const data = await response.json();
        setDatasets(data.datasets);
      } else {
        throw new Error('获取数据集失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取数据集失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchDataVersions = async (datasetName: string) => {
    try {
      setLoading(true);
      const response = await fetch(`/api/v1/training-data/datasets/${datasetName}/versions`);
      if (response.ok) {
        const data = await response.json();
        setDataVersions(data.versions);
      } else {
        throw new Error('获取版本列表失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取版本列表失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await fetch('/api/v1/training-data/statistics');
      if (response.ok) {
        const data = await response.json();
        setStatistics(data);
      }
    } catch (err) {
      console.error('获取统计信息失败:', err);
    }
  };

  // 创建数据源
  const handleCreateDataSource = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/training-data/sources', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newDataSource)
      });
      
      if (response.ok) {
        setDataSourceDialog(false);
        setNewDataSource({
          source_id: '',
          source_type: 'file',
          name: '',
          description: '',
          config: {}
        });
        await fetchDataSources();
      } else {
        throw new Error('创建数据源失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建数据源失败');
    } finally {
      setLoading(false);
    }
  };

  // 创建标注任务
  const handleCreateAnnotationTask = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/training-data/annotation-tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newTask)
      });
      
      if (response.ok) {
        setTaskDialog(false);
        setNewTask({
          name: '',
          description: '',
          task_type: 'text_classification',
          data_records: [],
          annotation_schema: {},
          guidelines: '',
          assignees: [],
          created_by: 'user_001'
        });
        await fetchAnnotationTasks();
      } else {
        throw new Error('创建标注任务失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建标注任务失败');
    } finally {
      setLoading(false);
    }
  };

  // 创建数据版本
  const handleCreateVersion = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/training-data/versions?created_by=user_001', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newVersion)
      });
      
      if (response.ok) {
        setVersionDialog(false);
        setNewVersion({
          dataset_name: '',
          version_number: '',
          description: '',
          data_record_ids: [],
          parent_version: '',
          metadata: {}
        });
        await fetchDatasets();
        if (selectedDataset) {
          await fetchDataVersions(selectedDataset);
        }
      } else {
        throw new Error('创建版本失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建版本失败');
    } finally {
      setLoading(false);
    }
  };

  // 触发数据收集
  const handleCollectData = async (sourceId: string) => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/training-data/collect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_id: sourceId,
          preprocessing_config: {
            rules: ['text_cleaning', 'format_standardization', 'quality_filtering']
          }
        })
      });
      
      if (response.ok) {
        alert('数据收集任务已启动');
        setTimeout(() => fetchDataRecords(), 2000);
      } else {
        throw new Error('启动数据收集失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动数据收集失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (currentTab === 0) {
      fetchDataSources();
      fetchStatistics();
    } else if (currentTab === 1) {
      fetchDataRecords();
    } else if (currentTab === 2) {
      fetchAnnotationTasks();
    } else if (currentTab === 3) {
      fetchDatasets();
    }
  }, [currentTab, recordsPage, recordsPerPage, recordsFilter]);

  useEffect(() => {
    if (selectedDataset) {
      fetchDataVersions(selectedDataset);
    }
  }, [selectedDataset]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'processed': return 'success';
      case 'raw': return 'warning';
      case 'error': return 'error';
      case 'validated': return 'info';
      default: return 'default';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        训练数据管理系统
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
          <Tab label="数据管理总览" />
          <Tab label="数据源管理" />
          <Tab label="数据收集" />
          <Tab label="数据预处理" />
          <Tab label="数据标注管理" />
          <Tab label="标注任务" />
          <Tab label="标注质量控制" />
          <Tab label="数据版本管理" />
          <Tab label="版本对比分析" />
          <Tab label="数据导出工具" />
          <Tab label="数据统计分析" />
          <Tab label="质量指标监控" />
        </Tabs>
      </Box>

      {/* 数据管理总览 */}
      <TabPanel value={currentTab} index={0}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6">数据管理总览</Typography>
        </Box>
        
        {statistics && (
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary">
                    {statistics.total_records || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    总数据记录
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="success.main">
                    {statistics.status_distribution?.processed || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    已处理记录
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="info.main">
                    {statistics.quality_stats?.average ? (statistics.quality_stats.average * 100).toFixed(1) : '0.0'}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    平均质量分数
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="warning.main">
                    {dataSources.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    数据源数量
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}
        
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              系统运行状态
            </Typography>
            <Alert severity="success" sx={{ mb: 1 }}>
              数据收集服务：运行正常
            </Alert>
            <Alert severity="success" sx={{ mb: 1 }}>
              标注管理服务：运行正常
            </Alert>
            <Alert severity="success">
              版本管理服务：运行正常
            </Alert>
          </CardContent>
        </Card>
      </TabPanel>

      {/* 数据源管理 */}
      <TabPanel value={currentTab} index={1}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h6">数据源列表</Typography>
          <Box>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={fetchDataSources}
              sx={{ mr: 2 }}
            >
              刷新
            </Button>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setDataSourceDialog(true)}
            >
              添加数据源
            </Button>
          </Box>
        </Box>

        {statistics && (
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary">
                    {statistics.total_records}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    总数据记录
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="success.main">
                    {statistics.status_distribution?.processed || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    已处理记录
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="info.main">
                    {(statistics.quality_stats?.average * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    平均质量分数
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="warning.main">
                    {statistics.status_distribution?.raw || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    待处理记录
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}

        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>名称</TableCell>
                <TableCell>类型</TableCell>
                <TableCell>描述</TableCell>
                <TableCell>状态</TableCell>
                <TableCell>创建时间</TableCell>
                <TableCell>操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {dataSources.map((source) => (
                <TableRow key={source.id}>
                  <TableCell>{source.name}</TableCell>
                  <TableCell>{source.source_type}</TableCell>
                  <TableCell>{source.description}</TableCell>
                  <TableCell>
                    <Chip
                      label={source.is_active ? '活跃' : '非活跃'}
                      color={source.is_active ? 'success' : 'default'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {new Date(source.created_at).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    <Tooltip title="收集数据">
                      <IconButton
                        size="small"
                        color="primary"
                        onClick={() => handleCollectData(source.source_id)}
                        disabled={!source.is_active}
                      >
                        <DownloadIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="编辑">
                      <IconButton size="small" color="default">
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* 数据收集 */}
      <TabPanel value={currentTab} index={2}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h6">数据收集</Typography>
          <Button
            variant="contained"
            startIcon={<UploadIcon />}
            onClick={() => {/* 启动收集任务 */}}
          >
            启动收集
          </Button>
        </Box>
        
        <Alert severity="info" sx={{ mb: 3 }}>
          从已配置的数据源中收集和预处理数据，自动进行质量评估。
        </Alert>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  收集任务进度
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    当前任务：API数据源收集
                  </Typography>
                  <LinearProgress variant="determinate" value={67} sx={{ mt: 1 }} />
                  <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                    已收集: 1,245 / 1,856 条记录 (67%)
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    预计完成时间：约 15 分钟
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  收集统计
                </Typography>
                <Typography variant="body2">今日收集: 2,456 条</Typography>
                <Typography variant="body2">本周收集: 18,923 条</Typography>
                <Typography variant="body2">总收集量: 156,789 条</Typography>
                <Typography variant="body2" color="error">
                  失败任务: 3 个
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* 数据预处理 */}
      <TabPanel value={currentTab} index={3}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h6">数据预处理</Typography>
          <Button
            variant="contained"
            startIcon={<SettingOutlined />}
            onClick={() => {/* 配置预处理规则 */}}
          >
            配置规则
          </Button>
        </Box>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  预处理规则
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Chip label="文本清理" color="success" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="去重" color="success" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="格式标准化" color="success" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="质量评估" color="success" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="语言检测" color="success" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="情感分析" color="warning" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="实体抽取" color="default" sx={{ mr: 1, mb: 1 }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  处理统计
                </Typography>
                <Typography variant="body2">处理队列: 245 条</Typography>
                <Typography variant="body2">处理速度: 120 条/分钟</Typography>
                <Typography variant="body2">平均质量分: 0.87</Typography>
                <Typography variant="body2" color="success.main">
                  成功率: 98.5%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* 数据标注管理 */}
      <TabPanel value={currentTab} index={4}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h6">数据标注管理</Typography>
          <Button
            variant="contained"
            startIcon={<EditIcon />}
            onClick={() => setTaskDialog(true)}
          >
            创建标注任务
          </Button>
        </Box>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  标注进度概览
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>任务类型</TableCell>
                        <TableCell>总数</TableCell>
                        <TableCell>已标注</TableCell>
                        <TableCell>进度</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>文本分类</TableCell>
                        <TableCell>1,245</TableCell>
                        <TableCell>856</TableCell>
                        <TableCell>
                          <LinearProgress variant="determinate" value={68.8} />
                          <Typography variant="caption">68.8%</Typography>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>序列标注</TableCell>
                        <TableCell>892</TableCell>
                        <TableCell>234</TableCell>
                        <TableCell>
                          <LinearProgress variant="determinate" value={26.2} />
                          <Typography variant="caption">26.2%</Typography>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  标注者状态
                </Typography>
                <Typography variant="body2">在线标注者: 8 人</Typography>
                <Typography variant="body2">今日完成: 456 条</Typography>
                <Typography variant="body2">平均用时: 2.3 分钟/条</Typography>
                <Typography variant="body2" color="success.main">
                  质量评分: 4.2/5.0
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* 数据记录 */}
      <TabPanel value={currentTab} index={1}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h6">数据记录</Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchDataRecords}
          >
            刷新
          </Button>
        </Box>

        {/* 过滤器 */}
        <Paper sx={{ p: 2, mb: 3 }}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <FormControl fullWidth size="small">
                <InputLabel>数据源</InputLabel>
                <Select
                  value={recordsFilter.source_id}
                  label="数据源"
                  onChange={(e) => setRecordsFilter(prev => ({ ...prev, source_id: e.target.value }))}
                >
                  <MenuItem value="">全部</MenuItem>
                  {dataSources.map((source) => (
                    <MenuItem key={source.source_id} value={source.source_id}>
                      {source.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <FormControl fullWidth size="small">
                <InputLabel>状态</InputLabel>
                <Select
                  value={recordsFilter.status}
                  label="状态"
                  onChange={(e) => setRecordsFilter(prev => ({ ...prev, status: e.target.value }))}
                >
                  <MenuItem value="">全部</MenuItem>
                  <MenuItem value="raw">原始</MenuItem>
                  <MenuItem value="processed">已处理</MenuItem>
                  <MenuItem value="validated">已验证</MenuItem>
                  <MenuItem value="rejected">已拒绝</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                size="small"
                label="最低质量分数"
                type="number"
                value={recordsFilter.min_quality_score}
                onChange={(e) => setRecordsFilter(prev => ({ ...prev, min_quality_score: e.target.value }))}
                inputProps={{ min: 0, max: 1, step: 0.1 }}
              />
            </Grid>
          </Grid>
        </Paper>

        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>记录ID</TableCell>
                <TableCell>数据源</TableCell>
                <TableCell>状态</TableCell>
                <TableCell>质量分数</TableCell>
                <TableCell>创建时间</TableCell>
                <TableCell>处理时间</TableCell>
                <TableCell>操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {dataRecords.map((record) => (
                <TableRow key={record.id}>
                  <TableCell>{record.record_id.substring(0, 8)}...</TableCell>
                  <TableCell>{record.source_id}</TableCell>
                  <TableCell>
                    <Chip
                      label={record.status}
                      color={getStatusColor(record.status) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {record.quality_score ? (
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <LinearProgress
                          variant="determinate"
                          value={record.quality_score * 100}
                          sx={{ width: 60, mr: 1 }}
                        />
                        <Typography variant="body2">
                          {(record.quality_score * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    ) : (
                      '-'
                    )}
                  </TableCell>
                  <TableCell>
                    {new Date(record.created_at).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    {record.processed_at 
                      ? new Date(record.processed_at).toLocaleString()
                      : '-'
                    }
                  </TableCell>
                  <TableCell>
                    <Tooltip title="查看详情">
                      <IconButton size="small" color="primary">
                        <VisibilityIcon />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <TablePagination
            component="div"
            count={-1}
            page={recordsPage}
            onPageChange={(e, newPage) => setRecordsPage(newPage)}
            rowsPerPage={recordsPerPage}
            onRowsPerPageChange={(e) => setRecordsPerPage(parseInt(e.target.value, 10))}
            rowsPerPageOptions={[5, 10, 25]}
          />
        </TableContainer>
      </TabPanel>

      {/* 标注任务 */}
      <TabPanel value={currentTab} index={5}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h6">标注任务</Typography>
          <Box>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={fetchAnnotationTasks}
              sx={{ mr: 2 }}
            >
              刷新
            </Button>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setTaskDialog(true)}
            >
              创建任务
            </Button>
          </Box>
        </Box>

        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>任务名称</TableCell>
                <TableCell>类型</TableCell>
                <TableCell>数据记录数</TableCell>
                <TableCell>分配人数</TableCell>
                <TableCell>状态</TableCell>
                <TableCell>创建时间</TableCell>
                <TableCell>截止时间</TableCell>
                <TableCell>操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {annotationTasks.map((task) => (
                <TableRow key={task.id}>
                  <TableCell>{task.name}</TableCell>
                  <TableCell>{task.task_type}</TableCell>
                  <TableCell>{task.data_record_count}</TableCell>
                  <TableCell>{task.assignee_count}</TableCell>
                  <TableCell>
                    <Chip
                      label={task.status}
                      color={task.status === 'completed' ? 'success' : 'default'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {new Date(task.created_at).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    {task.deadline 
                      ? new Date(task.deadline).toLocaleString()
                      : '-'
                    }
                  </TableCell>
                  <TableCell>
                    <Tooltip title="查看进度">
                      <IconButton size="small" color="primary">
                        <AssessmentIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="编辑">
                      <IconButton size="small" color="default">
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* 数据版本管理 */}
      <TabPanel value={currentTab} index={7}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h6">版本管理</Typography>
          <Box>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={fetchDatasets}
              sx={{ mr: 2 }}
            >
              刷新
            </Button>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setVersionDialog(true)}
            >
              创建版本
            </Button>
          </Box>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                数据集列表
              </Typography>
              {datasets.map((dataset) => (
                <Card
                  key={dataset.dataset_name}
                  sx={{ 
                    mb: 1, 
                    cursor: 'pointer',
                    bgcolor: selectedDataset === dataset.dataset_name ? 'action.selected' : 'background.paper'
                  }}
                  onClick={() => setSelectedDataset(dataset.dataset_name)}
                >
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Typography variant="body1" gutterBottom>
                      {dataset.dataset_name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      版本数: {dataset.version_count}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      记录数: {dataset.total_records}
                    </Typography>
                  </CardContent>
                </Card>
              ))}
            </Paper>
          </Grid>

          <Grid item xs={12} md={8}>
            {selectedDataset ? (
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  {selectedDataset} - 版本列表
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>版本号</TableCell>
                        <TableCell>记录数</TableCell>
                        <TableCell>大小</TableCell>
                        <TableCell>创建者</TableCell>
                        <TableCell>创建时间</TableCell>
                        <TableCell>操作</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {dataVersions.map((version) => (
                        <TableRow key={version.version_id}>
                          <TableCell>{version.version_number}</TableCell>
                          <TableCell>{version.record_count}</TableCell>
                          <TableCell>
                            {(version.size_bytes / 1024).toFixed(1)} KB
                          </TableCell>
                          <TableCell>{version.created_by}</TableCell>
                          <TableCell>
                            {new Date(version.created_at).toLocaleString()}
                          </TableCell>
                          <TableCell>
                            <Tooltip title="导出">
                              <IconButton size="small" color="primary">
                                <DownloadIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="删除">
                              <IconButton size="small" color="error">
                                <DeleteIcon />
                              </IconButton>
                            </Tooltip>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            ) : (
              <Paper sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="body1" color="text.secondary">
                  请选择一个数据集查看版本信息
                </Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      </TabPanel>

      {/* 标注质量控制 */}
      <TabPanel value={currentTab} index={6}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h6">标注质量控制</Typography>
          <Box>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={() => {/* 质量报告刷新 */}}
              sx={{ mr: 2 }}
            >
              刷新报告
            </Button>
            <Button
              variant="contained"
              startIcon={<AssessmentIcon />}
              onClick={() => {/* 生成质量报告 */}}
            >
              生成报告
            </Button>
          </Box>
        </Box>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  标注一致性分析
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Fleiss' Kappa: 0.75 (良好)
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  整体一致性率: 82.3%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  冲突标注数: 45 条
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  标注者绩效统计
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  平均标注时间: 2.3 分钟/条
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  质量评分: 4.2/5.0
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  活跃标注者: 12 人
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* 版本对比分析 */}
      <TabPanel value={currentTab} index={8}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h6">版本对比分析</Typography>
          <Button
            variant="contained"
            startIcon={<DiffOutlined />}
            onClick={() => {/* 版本比较 */}}
          >
            比较版本
          </Button>
        </Box>
        
        <Alert severity="info" sx={{ mb: 3 }}>
          选择两个版本进行详细对比分析，查看数据变化和质量差异。
        </Alert>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  版本 A: v1.2.0
                </Typography>
                <Typography variant="body2">记录数: 10,245</Typography>
                <Typography variant="body2">平均质量分: 0.87</Typography>
                <Typography variant="body2">创建时间: 2024-01-15</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  版本 B: v1.3.0
                </Typography>
                <Typography variant="body2">记录数: 12,156</Typography>
                <Typography variant="body2">平均质量分: 0.91</Typography>
                <Typography variant="body2">创建时间: 2024-01-20</Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* 数据导出工具 */}
      <TabPanel value={currentTab} index={9}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h6">数据导出工具</Typography>
        </Box>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  导出格式
                </Typography>
                <FormControl fullWidth margin="normal">
                  <InputLabel>选择格式</InputLabel>
                  <Select defaultValue="jsonl">
                    <MenuItem value="jsonl">JSONL</MenuItem>
                    <MenuItem value="json">JSON</MenuItem>
                    <MenuItem value="csv">CSV</MenuItem>
                    <MenuItem value="parquet">Parquet</MenuItem>
                  </Select>
                </FormControl>
                <Button
                  fullWidth
                  variant="contained"
                  startIcon={<DownloadIcon />}
                  sx={{ mt: 2 }}
                >
                  开始导出
                </Button>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  导出历史
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>文件名</TableCell>
                        <TableCell>格式</TableCell>
                        <TableCell>大小</TableCell>
                        <TableCell>状态</TableCell>
                        <TableCell>导出时间</TableCell>
                        <TableCell>操作</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>training_data_v1.2.jsonl</TableCell>
                        <TableCell>JSONL</TableCell>
                        <TableCell>45.2 MB</TableCell>
                        <TableCell><Chip label="完成" color="success" size="small" /></TableCell>
                        <TableCell>2024-01-20 14:30</TableCell>
                        <TableCell>
                          <IconButton size="small" color="primary">
                            <DownloadIcon />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* 数据统计分析 */}
      <TabPanel value={currentTab} index={10}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h6">数据统计分析</Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => {/* 刷新统计 */}}
          >
            刷新统计
          </Button>
        </Box>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  数据分布统计
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2">文本分类: 45.2%</Typography>
                  <LinearProgress variant="determinate" value={45.2} sx={{ mt: 1, mb: 1 }} />
                  <Typography variant="body2">序列标注: 28.7%</Typography>
                  <LinearProgress variant="determinate" value={28.7} sx={{ mt: 1, mb: 1 }} />
                  <Typography variant="body2">问答对话: 26.1%</Typography>
                  <LinearProgress variant="determinate" value={26.1} sx={{ mt: 1, mb: 1 }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  质量趋势分析
                </Typography>
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body2" color="text.secondary">
                    质量趋势图表
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    （图表组件将在实际应用中实现）
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* 质量指标监控 */}
      <TabPanel value={currentTab} index={11}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h6">质量指标监控</Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => {/* 刷新监控 */}}
          >
            实时刷新
          </Button>
        </Box>
        
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="primary">
                  89.2%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  整体数据质量
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="success.main">
                  2.1s
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  平均处理时间
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="warning.main">
                  156
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  待处理队列
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="error.main">
                  3
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  异常告警
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              实时监控日志
            </Typography>
            <Alert severity="success" sx={{ mb: 1 }}>
              [14:23:45] 数据预处理任务完成 - 处理记录数: 245
            </Alert>
            <Alert severity="warning" sx={{ mb: 1 }}>
              [14:22:12] 检测到质量异常数据 - 记录ID: rec_8847
            </Alert>
            <Alert severity="info">
              [14:21:08] 新数据源已连接 - 来源: API-001
            </Alert>
          </CardContent>
        </Card>
      </TabPanel>

      {/* 创建数据源对话框 */}
      <Dialog open={dataSourceDialog} onClose={() => setDataSourceDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>创建数据源</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            margin="normal"
            label="数据源ID"
            value={newDataSource.source_id}
            onChange={(e) => setNewDataSource(prev => ({ ...prev, source_id: e.target.value }))}
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>数据源类型</InputLabel>
            <Select
              value={newDataSource.source_type}
              label="数据源类型"
              onChange={(e) => setNewDataSource(prev => ({ ...prev, source_type: e.target.value }))}
            >
              <MenuItem value="file">文件</MenuItem>
              <MenuItem value="api">API</MenuItem>
              <MenuItem value="web">网页</MenuItem>
              <MenuItem value="database">数据库</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            margin="normal"
            label="数据源名称"
            value={newDataSource.name}
            onChange={(e) => setNewDataSource(prev => ({ ...prev, name: e.target.value }))}
          />
          <TextField
            fullWidth
            margin="normal"
            label="描述"
            multiline
            rows={3}
            value={newDataSource.description}
            onChange={(e) => setNewDataSource(prev => ({ ...prev, description: e.target.value }))}
          />
          <TextField
            fullWidth
            margin="normal"
            label="配置 (JSON)"
            multiline
            rows={4}
            value={JSON.stringify(newDataSource.config, null, 2)}
            onChange={(e) => {
              try {
                const config = JSON.parse(e.target.value);
                setNewDataSource(prev => ({ ...prev, config }));
              } catch {
                // 忽略无效JSON
              }
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDataSourceDialog(false)}>取消</Button>
          <Button 
            onClick={handleCreateDataSource} 
            variant="contained"
            disabled={loading || !newDataSource.source_id || !newDataSource.name}
          >
            {loading ? <CircularProgress size={20} /> : '创建'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* 创建标注任务对话框 */}
      <Dialog open={taskDialog} onClose={() => setTaskDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>创建标注任务</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            margin="normal"
            label="任务名称"
            value={newTask.name}
            onChange={(e) => setNewTask(prev => ({ ...prev, name: e.target.value }))}
          />
          <TextField
            fullWidth
            margin="normal"
            label="任务描述"
            multiline
            rows={3}
            value={newTask.description}
            onChange={(e) => setNewTask(prev => ({ ...prev, description: e.target.value }))}
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>任务类型</InputLabel>
            <Select
              value={newTask.task_type}
              label="任务类型"
              onChange={(e) => setNewTask(prev => ({ ...prev, task_type: e.target.value }))}
            >
              <MenuItem value="text_classification">文本分类</MenuItem>
              <MenuItem value="sequence_labeling">序列标注</MenuItem>
              <MenuItem value="question_answering">问答</MenuItem>
              <MenuItem value="sentiment_analysis">情感分析</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            margin="normal"
            label="标注指南"
            multiline
            rows={4}
            value={newTask.guidelines}
            onChange={(e) => setNewTask(prev => ({ ...prev, guidelines: e.target.value }))}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTaskDialog(false)}>取消</Button>
          <Button 
            onClick={handleCreateAnnotationTask} 
            variant="contained"
            disabled={loading || !newTask.name}
          >
            {loading ? <CircularProgress size={20} /> : '创建'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* 创建版本对话框 */}
      <Dialog open={versionDialog} onClose={() => setVersionDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>创建数据版本</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            margin="normal"
            label="数据集名称"
            value={newVersion.dataset_name}
            onChange={(e) => setNewVersion(prev => ({ ...prev, dataset_name: e.target.value }))}
          />
          <TextField
            fullWidth
            margin="normal"
            label="版本号"
            value={newVersion.version_number}
            onChange={(e) => setNewVersion(prev => ({ ...prev, version_number: e.target.value }))}
          />
          <TextField
            fullWidth
            margin="normal"
            label="版本描述"
            multiline
            rows={3}
            value={newVersion.description}
            onChange={(e) => setNewVersion(prev => ({ ...prev, description: e.target.value }))}
          />
          <TextField
            fullWidth
            margin="normal"
            label="父版本ID (可选)"
            value={newVersion.parent_version}
            onChange={(e) => setNewVersion(prev => ({ ...prev, parent_version: e.target.value }))}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setVersionDialog(false)}>取消</Button>
          <Button 
            onClick={handleCreateVersion} 
            variant="contained"
            disabled={loading || !newVersion.dataset_name || !newVersion.version_number}
          >
            {loading ? <CircularProgress size={20} /> : '创建'}
          </Button>
        </DialogActions>
      </Dialog>

      {loading && (
        <Box sx={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 9999 }}>
          <LinearProgress />
        </Box>
      )}
    </Box>
  );
}